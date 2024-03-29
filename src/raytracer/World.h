#pragma once

#include "Hittable.h"
#include "AABB.h"

#include "thrust/device_ptr.h"
#include "thrust/device_malloc.h"
#include "thrust/device_free.h"



class World : public Hittable {
public:
    int number_of_objects;
    int capacity;
    Hittable** objects;
    
    __device__ World() {
        objects = new Hittable*[20];
        number_of_objects = 0;
        capacity = 20;
    }

    __device__ World(int capacity) {
        objects = new Hittable*[capacity];
        number_of_objects = 0;
        capacity = capacity;
    }

    __device__ ~World();

    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        HitRecord temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;
        for (int i = 0; i < number_of_objects; i++) {
            if (objects[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }

    __device__ virtual bool bounding_box(float time0, float time1, AABB& output_box) const;

    __device__ bool add(Hittable* object) {
        if (number_of_objects >= capacity) {
			return false;
		}

        objects[number_of_objects] = object;
		number_of_objects++;
		return true;
    }

};

/*
* This is a pretty bad way to do this, but I honestly don't know how to do it better
* During the compilation it is checked if the code is being compiled by nvcc
* If it is, then the destructor is defined
* If it is not, then the destructor is not defined to avoid multiple definitions
*/ 

#ifdef __CUDACC__

#include "Sphere.h"

__device__ World::~World() {
    for (int i = 0; i < number_of_objects; i++) {
        delete (Sphere*) objects[i];
    }
    delete[] objects;
}

__device__ AABB surrounding_box(AABB box0, AABB box1) {
    glm::vec3 small(fmin(box0.min().x, box1.min().x),
        fmin(box0.min().y, box1.min().y),
        fmin(box0.min().z, box1.min().z));

    glm::vec3 big(fmax(box0.max().x, box1.max().x),
        fmax(box0.max().y, box1.max().y),
        fmax(box0.max().z, box1.max().z));

    return AABB(small, big);
}

__device__ bool World::bounding_box(float time0, float time1, AABB& output_box) const {
    if (number_of_objects < 1) {
        return false;
    }

    AABB temp_box;
    bool first_box = true;

    for (int i = 0; i < number_of_objects; i++) {
        if (!objects[i]->bounding_box(time0, time1, temp_box)) {
            return false;
        }
        output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
        first_box = false;
    }

    return true;
}

#endif