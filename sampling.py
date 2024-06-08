import mitsuba as mi
import drjit as dr



def ray_marching(mode, scene, sampler,
        ray, δL, state_in, active, **kwargs):
    primal = mode == dr.ADMode.Primal

    ray = mi.Ray3f(ray)
    hit, mint, maxt = self.bbox.ray_intersect(ray)

    active = mi.Bool(active)
    active &= hit  # ignore rays that miss the bbox
    if not primal:  # if the gradient is zero, stop early
        active &= dr.any(dr.neq(δL, 0))

    step_size = mi.Float(1.0 / self.grid_res)
    import ipdb; ipdb.set_trace()
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