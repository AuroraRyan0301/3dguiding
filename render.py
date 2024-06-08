
def random_spawn_ray(sensors, ray, batch_size, spp, seed):



def my_render(scene, sensors, integrator=None, spp=128, ref_images=None, save_path=None, seed = 0):
    # Camera origin in world space
    cam_origin = mi.Point3f(0, 1, 3)

    # Camera view direction in world space
    cam_dir = dr.normalize(mi.Vector3f(0, -0.5, -1))

    # Camera width and height in world space
    cam_width  = 2.0
    cam_height = 2.0

    # Image pixel resolution
    image_res = [256, 256]
