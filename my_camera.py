import mitsuba as mi
import drjit as dr

class OrthogonalCamera():
    
    # cam_origin: mi.Point3f
    # # camera position
    # cam_dir: mi.Vector3f
    # # camera direction
    # cam_width: float
    # cam_height: float
    # # Camera width and height in world space
    # image_res: list
    # # Image pixel resolution


    def __init__(self, cam_origin, cam_dir, cam_width=1.0, cam_height=1.0, image_res=[256, 256]):
        self.cam_origin = cam_origin
        self.cam_dir = cam_dir
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.image_res = image_res

    def spawn_image_rays(self):
        # Construct a grid of 2D coordinates
        x, y = dr.meshgrid(
            dr.linspace(mi.Float, -self.cam_width   / 2,   self.cam_width  / 2, self.image_res[0]),
            dr.linspace(mi.Float, -self.cam_height  / 2, self.cam_height  / 2, self.image_res[1])
        )
        # it's from left to right, top to bottom

        # Ray origin in local coordinates
        ray_origin_local = mi.Vector3f(x, y, 0)

        # Ray origin in world coordinates
        ray_origin = mi.Frame3f(self.cam_dir).to_world(ray_origin_local) + self.cam_origin

        ray = mi.Ray3f(o=ray_origin, d=self.cam_dir)

        return ray
