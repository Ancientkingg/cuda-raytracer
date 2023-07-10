#pragma once

#include "Hittable.h"
#include "AABB.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/sort.h>
#include "World.h"

struct BoxCmp {
    __device__ BoxCmp(int axis) : axis(axis) {}

    __device__ bool operator()(Hittable* a, Hittable* b) {
		AABB box_left, box_right;

        if (!a->bounding_box(0, 0, box_left) || !b->bounding_box(0, 0, box_right)) {
            printf("No bounding box in bvh_node constructor.\n");
            return false;
        }

        float left_min, right_min;

        if (axis == 1) {
            left_min = box_left.min().x;
            right_min = box_right.min().x;
        } else if (axis == 2) {
			left_min = box_left.min().y;
			right_min = box_right.min().y;
		} else if (axis == 3) {
			left_min = box_left.min().z;
			right_min = box_right.min().z;
		}

        return left_min < right_min;
	}

    // Axis: 1 = x, 2 = y, 3 = z
    int axis;
};

__device__ AABB surrounding_box(AABB box0, AABB box1);

class BVHNode : public Hittable {
public:
	__device__ BVHNode();

    __device__ BVHNode(Hittable** objects, int n, float time0, float time1, curandState* local_rand_state) {
        int axis = int(3 * curand_uniform(local_rand_state));

        if (axis == 0) {
            thrust::sort(objects, objects + n, BoxCmp(1));
        }
        else if (axis == 1) {
            thrust::sort(objects, objects + n, BoxCmp(2));
        } else {
            thrust::sort(objects, objects + n, BoxCmp(3));
        }

        if (n == 1) {
            left = right = objects[0];
        } else if (n == 2) {
            left = objects[0];
            right = objects[1];
        } else {
            left = new BVHNode(objects, n / 2, time0, time1, local_rand_state);
            right = new BVHNode(objects + n / 2, n - n / 2, time0, time1, local_rand_state);
        }

        AABB box_left, box_right;

        if (!left->bounding_box(time0, time1, box_left) ||
            !right->bounding_box(time0, time1, box_right)) {
                return;
        }

        box = surrounding_box(box_left, box_right);
    }

	__device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        if (box.hit(r, t_min, t_max)) {
            HitRecord left_rec, right_rec;
            bool hit_left = left->hit(r, t_min, t_max, left_rec);
            bool hit_right = right->hit(r, t_min, t_max, right_rec);
            if (hit_left && hit_right) {
                if (left_rec.t < right_rec.t)
                    rec = left_rec;
                else
                    rec = right_rec;
                return true;
            }
            else if (hit_left) {
                rec = left_rec;
                return true;
            }
            else if (hit_right) {
                rec = right_rec;
                return true;
            }
            else
                return false;
        }
        else return false;
	}
    __device__ virtual bool bounding_box(float t0, float t1, AABB& output_box) const {
        output_box = box;
        return true;
    }

	Hittable* left;
	Hittable* right;
	AABB box;

    __device__ ~BVHNode() {
        delete left;
        delete right;
    }
};