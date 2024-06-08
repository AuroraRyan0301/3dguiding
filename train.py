import mitsuba as mi
import drjit as dr
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')

from config import *
from my_integrator import *
import matplotlib.pyplot as plt
import os

def get_random_mask(batch_num, res = RENDER_RES):
    total_points = res * res
    point_indices = np.random.choice(total_points, size=batch_num, replace=False)

    mask = np.zeros(res*res, dtype=int)
    mask.flat[point_indices] = 1
    image_mask = mask.reshape(res, res)
    # replicate 4 channel
    image_mask = np.repeat(image_mask[:, :, np.newaxis], 4, axis=2)
    return mi.Bool(mask),image_mask


def get_voxel_importance_weights(sigmat_grad):
    abs_grad = dr.abs(sigmat_grad)
    res = abs_grad.shape[0]
    vertex_1 = abs_grad[0:res-1, 0:res-1, 0:res-1]
    vertex_2 = abs_grad[1:res, 0:res-1, 0:res-1]
    vertex_3 = abs_grad[0:res-1, 1:res, 0:res-1]
    vertex_4 = abs_grad[0:res-1, 0:res-1, 1:res]
    vertex_5 = abs_grad[1:res, 1:res, 0:res-1]
    vertex_6 = abs_grad[1:res, 0:res-1, 1:res]
    vertex_7 = abs_grad[0:res-1, 1:res, 1:res]
    vertex_8 = abs_grad[1:res, 1:res, 1:res]
    voxel_importance = (vertex_1 + vertex_2 + vertex_3 + \
                         vertex_4 + vertex_5 + vertex_6 + \
                            vertex_7 + vertex_8) / 8
    voxel_importance += 1e-11
    voxel_importance_sum = dr.sum(voxel_importance)
    normalized_voxel_importance = voxel_importance / dr.maximum(voxel_importance_sum, 1e-11)

    return normalized_voxel_importance

def train(scene, integrator, sensors, ref_images):
    print("training...")
    params = mi.traverse(integrator)
    opt = mi.ad.Adam(lr=LR, params={'sigmat': params['sigmat'], 'sh_coeffs': params['sh_coeffs']})
    params.update(opt)

    losses = []
    intermediate_images = []

    for stage in range(NUM_STAGES):
        print(f"Stage {stage+1:02d}, feature voxel grids resolution -> {opt['sigmat'].shape[0]}")
        for it in range(NUM_iterations_per_stage):
            total_loss = 0.0
            images = []
            if it == 0:
                for sensor_idx in range( SENSOR_COUNT):
                    img = mi.render(scene, params, sensor=sensors[sensor_idx], spp=1, seed=it)
                    loss = dr.mean(dr.abs(img - ref_images[sensor_idx]))
                    dr.backward(loss)
                    total_loss += loss[0]

                    # Store images at the end of every stage
                    if it == NUM_iterations_per_stage- 1:
                        dr.eval(img)
                        images.append(img)

                losses.append(total_loss)
                voxel_vertex_weight = get_voxel_importance_weights(dr.grad(opt['sigmat']))
                opt.step()
            else:
                assert voxel_vertex_weight is not None
                for sensor_idx in range( SENSOR_COUNT):
                    rand_mask, corresponding_img_mask = get_random_mask(batch_num=BATCH_NUM, res=RENDER_RES)
                    # import ipdb; ipdb.set_trace()
                    integrator.mask = rand_mask
                    img = mi.render(scene, params, sensor=sensors[sensor_idx], integrator = integrator, spp=1, seed=it)
                    integrator.mask = None
                    loss = dr.mean((dr.abs(img - ref_images[sensor_idx]) * corresponding_img_mask)*RENDER_RES*RENDER_RES/BATCH_NUM)
                    dr.backward(loss)
                    total_loss += loss[0]

                    # Store images at the end of every stage
                    if it == NUM_iterations_per_stage- 1:
                        dr.eval(img)
                        images.append(img)

                losses.append(total_loss)
                opt.step()

            if not integrator.use_relu:
                opt['sigmat'] = dr.maximum(opt['sigmat'], 0.0)

            params.update(opt)
            print(f"  --> iteration {it+1:02d}: error={total_loss:6f}", end='\r')

        intermediate_images.append(images)

        # Upsample the 3D textures at every stage
        if stage < NUM_STAGES - 1:
            new_res = 2 * opt['sigmat'].shape[0]
            new_shape = [new_res, new_res, new_res]
            opt['sigmat']   = dr.upsample(opt['sigmat'],   new_shape)
            opt['sh_coeffs'] = dr.upsample(opt['sh_coeffs'], new_shape)
            params.update(opt)

    print('')
    print('Done')
    return intermediate_images

def register_integrator():
    print("registering integrator...")
    mi.register_integrator("nerf", lambda props: RadianceFieldPRB(props))
    mi.register_integrator("mask_nerf", lambda props: MaskNeRFPRB(props))

def initialize_sensor(sensor_count=SENSOR_COUNT, render_res=RENDER_RES):
    print("initializing sensors...")
    sensors = []

    for i in range(sensor_count):
        angle = 360.0 / sensor_count * i
        sensors.append(mi.load_dict({
            'type': 'perspective',
            'fov': 45,
            'to_world': mi.ScalarTransform4f.translate([0.5, 0.5, 0.5]) \
                                            .rotate([0, 1, 0], angle)   \
                                            .look_at(target=[0, 0, 0],
                                                    origin=[0, 0, 1.3],
                                                    up=[0, 1, 0]),
            'film': {
                'type': 'hdrfilm',
                'width': render_res,
                'height': render_res,
                'filter': {'type': 'box'},
                'pixel_format': 'rgba'
            }
        }))
    
    return sensors

def plot_list(images, title=None, save_path=None):
    fig, axs = plt.subplots(1, len(images), figsize=(18, 3))
    for i in range(len(images)):
        axs[i].imshow(mi.util.convert_to_bitmap(images[i]))
        axs[i].axis('off')
    if title is not None:
        plt.suptitle(title)

def save_final_images(images, save_path):
    for i, img in enumerate(images):
        mi.util.write_bitmap(f'{save_path}/final_{i}.exr', img)

def create_ref_scene(sensors, sensor_count=SENSOR_COUNT):
    print("creating reference scene...")
    scene_ref = mi.load_file('./scenes/lego/scene.xml')
    ref_images = [mi.render(scene_ref, sensor=sensors[i], spp=64) for i in range(sensor_count)]
    # plot_list(ref_images, 'Reference images')
    return ref_images, scene_ref


def initialize_scene(integrator_name, sensors=None, sensor_count=SENSOR_COUNT):
    print("initializing scene...")
    scene = mi.load_dict({
    'type': 'scene',
    'integrator': {
        'type': integrator_name
    },
    'emitter': {
        'type': 'constant'
    }
    })
    integrator = scene.integrator()

    # # Render initial state
    # init_images = [mi.render(scene, sensor=sensors[i], spp=128) for i in range(sensor_count)]
    return scene, integrator

def show_final_images(intermediate_images,sensor_count=SENSOR_COUNT, ref_images=None, save_path=None):
    final_images = [mi.render(scene, sensor=sensors[i], spp=128) for i in range(sensor_count)]
    # for stage, inter in enumerate(intermediate_images):
    #     plot_list(inter, f'Stage {stage}')
    # plot_list(final_images, 'Final')
    # plot_list(ref_images, 'Reference')
    if save_path is not None:
        # make sure the save_path exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_final_images(final_images, save_path)
    

if __name__ == '__main__':
    register_integrator()
    sensors = initialize_sensor()
    # import ipdb; ipdb.set_trace()
    ref_images, scene_ref = create_ref_scene(sensors)
    scene,integrator = initialize_scene("mask_nerf")
    intermediate_images = train(scene, integrator, sensors, ref_images)
    show_final_images(intermediate_images, ref_images=ref_images, save_path='./output')
    # plot_list(ref_images, 'Reference images')
    # plot_list(integrator.sigmat.eval(), 'Final sigma_t')
    # plot_list(integrator.sh_coeffs.eval(), 'Final SH coefficients')