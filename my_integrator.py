import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
from config import *
import numpy as np

from contextlib import contextmanager
@contextmanager
def dr_no_jit(when=True):
    if when:
        dr.set_flag(dr.JitFlag.LoopRecord, False)
        dr.set_flag(dr.JitFlag.VCallRecord, False)
    try:
        yield
    finally:
        if when:
            dr.set_flag(dr.JitFlag.LoopRecord, True)
            dr.set_flag(dr.JitFlag.VCallRecord, True)


class RadianceFieldPRB(mi.ad.common.RBIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.bbox = mi.ScalarBoundingBox3f([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        self.use_relu = USE_RELU
        self.grid_res = GRID_INIT_RES
        # Initialize the 3D texture for the density and SH coefficients
        res = self.grid_res
        self.sigmat = mi.Texture3f(dr.full(mi.TensorXf, 0.01, shape=(res, res, res, 1)))
        self.sh_coeffs = mi.Texture3f(dr.full(mi.TensorXf, 0.1, shape=(res, res, res, 3 * (SH_DEGREE + 1) ** 2)))

    def eval_emission(self, pos, direction):
        spec = mi.Spectrum(0)
        sh_dir_coef = dr.sh_eval(direction, SH_DEGREE)
        sh_coeffs = self.sh_coeffs.eval(pos)
        for i, sh in enumerate(sh_dir_coef):
            spec += sh * mi.Spectrum(sh_coeffs[3 * i:3 * (i + 1)])
        return dr.clip(spec, 0.0, 1.0)

    def sample(self, mode, scene, sampler,
               ray, δL, state_in, active, **kwargs):
        primal = mode == dr.ADMode.Primal

        ray = mi.Ray3f(ray)
        hit, mint, maxt = self.bbox.ray_intersect(ray)
        print("hit", hit)
        print("active", active)
        import ipdb; ipdb.set_trace()

        active = mi.Bool(active)
        active &= hit  # ignore rays that miss the bbox
        if not primal:  # if the gradient is zero, stop early
            active &= dr.any(dr.neq(δL, 0))

        step_size = mi.Float(1.0 / self.grid_res)
        t = mi.Float(mint) + sampler.next_1d(active) * step_size
        L = mi.Spectrum(0.0 if primal else state_in)
        δL = mi.Spectrum(δL if δL is not None else 0)
        β = mi.Spectrum(1.0) # throughput

        loop = mi.Loop(name=f"PRB ({mode.name})",
                       state=lambda: (sampler, ray, L, t, δL, β, active))
        while loop(active):
            p = ray(t)
            with dr.resume_grad(when=not primal):
                sigmat = self.sigmat.eval(p)[0]
                if self.use_relu:
                    sigmat = dr.maximum(sigmat, 0.0)
                tr = dr.exp(-sigmat * step_size)
                # Evaluate the directionally varying emission (weighted by transmittance)
                Le = β * (1.0 - tr) * self.eval_emission(p, ray.d)

            β *= tr
            L = L + Le if primal else L - Le

            with dr.resume_grad(when=not primal):
                if not primal:
                    dr.backward_from(δL * (L * tr / dr.detach(tr) + Le))

            t += step_size
            active &= (t < maxt) & dr.any(dr.neq(β, 0.0))

        return L if primal else δL, mi.Bool(True), L

    def traverse(self, callback):
        callback.put_parameter("sigmat", self.sigmat.tensor(), mi.ParamFlags.Differentiable)
        callback.put_parameter('sh_coeffs', self.sh_coeffs.tensor(), mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        self.sigmat.set_tensor(self.sigmat.tensor())
        self.sh_coeffs.set_tensor(self.sh_coeffs.tensor())
        self.grid_res = self.sigmat.shape[0]

class MaskNeRFPRB(mi.ad.common.RBIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.bbox = mi.ScalarBoundingBox3f([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        self.use_relu = USE_RELU
        self.grid_res = GRID_INIT_RES
        # Initialize the 3D texture for the density and SH coefficients
        res = self.grid_res
        self.sigmat = mi.Texture3f(dr.full(mi.TensorXf, 0.01, shape=(res, res, res, 1)))
        self.sh_coeffs = mi.Texture3f(dr.full(mi.TensorXf, 0.1, shape=(res, res, res, 3 * (SH_DEGREE + 1) ** 2)))
        self.mask = None

    def eval_emission(self, pos, direction):
        spec = mi.Spectrum(0)
        sh_dir_coef = dr.sh_eval(direction, SH_DEGREE)
        sh_coeffs = self.sh_coeffs.eval(pos)
        for i, sh in enumerate(sh_dir_coef):
            spec += sh * mi.Spectrum(sh_coeffs[3 * i:3 * (i + 1)])
        return dr.clip(spec, 0.0, 1.0)

    def sample(self, mode, scene, sampler,
               ray, δL, state_in, active, **kwargs):
        with dr_no_jit(when = True):
            primal = mode == dr.ADMode.Primal
            ray = mi.Ray3f(ray)
            hit, mint, maxt = self.bbox.ray_intersect(ray)

            active = mi.Bool(active)
            active &= hit  # ignore rays that miss the bbox
            # ignore rays not hitting the mask
            if self.mask is not None:
                active &= self.mask
            if not primal:  # if the gradient is zero, stop early
                active &= dr.any(dr.neq(δL, 0))

            step_size = mi.Float(1.0 / self.grid_res)
            t = mi.Float(mint) + sampler.next_1d(active) * step_size
            # import ipdb; ipdb.set_trace()
            L = mi.Spectrum(0.0 if primal else state_in)
            δL = mi.Spectrum(δL if δL is not None else 0)
            β = mi.Spectrum(1.0) # throughput

            loop = mi.Loop(name=f"PRB ({mode.name})",
                        state=lambda: (sampler, ray, L, t, δL, β, active))
            while loop(active):
                p = ray(t)
                pos = ray(t)
                with dr.resume_grad(when=not primal):
                    sigmat = self.sigmat.eval(p)[0]
                    if self.use_relu:
                        sigmat = dr.maximum(sigmat, 0.0)
                    tr = dr.exp(-sigmat * step_size)
                    # Evaluate the directionally varying emission (weighted by transmittance)
                    Le = β * (1.0 - tr) * self.eval_emission(p, ray.d)
                β *= tr
                L = L + Le if primal else L - Le

                with dr.resume_grad(when=not primal):
                    if not primal:
                        dr.backward_from(δL * (L * tr / dr.detach(tr) + Le))

                t += step_size
                active &= (t < maxt) & dr.any(dr.neq(β, 0.0))

            return L if primal else δL, mi.Bool(True), L

    def traverse(self, callback):
        callback.put_parameter("sigmat", self.sigmat.tensor(), mi.ParamFlags.Differentiable)
        callback.put_parameter('sh_coeffs', self.sh_coeffs.tensor(), mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        self.sigmat.set_tensor(self.sigmat.tensor())
        self.sh_coeffs.set_tensor(self.sh_coeffs.tensor())
        self.grid_res = self.sigmat.shape[0]