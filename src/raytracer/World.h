#pragma once

#include "Hittable.h"

class World : public Hittable {
public:
	Hittable** list; // array of pointers to objects in the world
	int list_size; // amount of objects in the world

	__device__ World() {}
	__device__ World(Hittable** l, int n) { list = l; list_size = n; }
    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        HitRecord temp_rec;
        bool hitAnything = false;
        float closest_so_far = t_max;
        for (int i = 0; i < list_size; i++) {
            if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hitAnything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hitAnything;
    }
};