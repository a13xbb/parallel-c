#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define DT 0.05

typedef struct
{
    double x, y;
} vector;

int bodies, timeSteps;
double *masses, GravConstant;
vector *positions, *velocities, *accelerations;

vector addVectors(vector a, vector b)
{
    vector c = {a.x + b.x, a.y + b.y};

    return c;
}

vector scaleVector(double b, vector a)
{
    vector c = {b * a.x, b * a.y};

    return c;
}

vector subtractVectors(vector a, vector b)
{
    vector c = {a.x - b.x, a.y - b.y};

    return c;
}

double mod(vector a)
{
    return sqrt(a.x * a.x + a.y * a.y);
}

void initiateSystem(char *fileName)
{
    int i;
    FILE *fp = fopen(fileName, "r");

    fscanf(fp, "%lf%d%d", &GravConstant, &bodies, &timeSteps);

    masses = (double *)malloc(bodies * sizeof(double));
    positions = (vector *)malloc(bodies * sizeof(vector));
    velocities = (vector *)malloc(bodies * sizeof(vector));
    accelerations = (vector *)malloc(bodies * sizeof(vector));

    for (i = 0; i < bodies; i++)
    {
        fscanf(fp, "%lf", &masses[i]);
        fscanf(fp, "%lf%lf", &positions[i].x, &positions[i].y);
        fscanf(fp, "%lf%lf", &velocities[i].x, &velocities[i].y);
    }

    fclose(fp);
}

void resolveCollisions()
{
    int i, j;

    for (i = 0; i < bodies - 1; i++)
        for (j = i + 1; j < bodies; j++)
        {
            if (positions[i].x == positions[j].x && positions[i].y == positions[j].y)
            {
                vector temp = velocities[i];
                velocities[i] = velocities[j];
                velocities[j] = temp;
            }
        }
}

void computeAccelerations()
{
    int my_rank = omp_get_thread_num();
    // printf("Computing in thread %d\n", my_rank);

    int thread_count = omp_get_num_threads();
    // printf("Summary threads: %d\n", thread_count);

    int chunk_sz = bodies / thread_count;

    int start = my_rank * chunk_sz;
    int end = start + chunk_sz;
    if (end > bodies) {
        end = bodies;
    }

    int i, j;

    for (i = start; i < end; i++) {
        accelerations[i].x = 0;
        accelerations[i].y = 0;
    }

    for (i = start; i < end; i++)
    {
        for (j = i + 1; j < end; j++)
        {
            // printf("calculating points %d and %d\n", i, j);
            vector acc_delta = scaleVector(GravConstant * 1 / pow(mod(subtractVectors(positions[i], positions[j])), 3), subtractVectors(positions[j], positions[i]));
            accelerations[i] = addVectors(accelerations[i], scaleVector(masses[j], acc_delta));
            accelerations[j] = addVectors(accelerations[j], scaleVector(-masses[i], acc_delta));
        }
    }
    
}

void computeVelocities()
{
    int i;

    for (i = 0; i < bodies; i++)
        velocities[i] = addVectors(velocities[i], scaleVector(DT, accelerations[i]));
}

void computePositions()
{
    int i;

    for (i = 0; i < bodies; i++)
        positions[i] = addVectors(positions[i], scaleVector(DT,velocities[i]));
}

void simulate()
{
#pragma omp parallel num_threads(3)
    computeAccelerations();
    computePositions();
    computeVelocities();
    resolveCollisions();
}

int main(int argC, char *argV[])
{
    int i, j;

    if (argC != 2)
        printf("Usage : %s <file name containing system configuration data>", argV[0]);
    else
    {
        initiateSystem(argV[1]);
        printf("Body   :     x              y           vx              vy   ");
        for (i = 0; i < timeSteps; i++)
        {
            printf("\nCycle %d\n", i + 1);
            simulate();
            for (j = 0; j < bodies; j++)
                printf("Body %d : %lf\t%lf\t%lf\t%lf\n", j + 1, positions[j].x, positions[j].y, velocities[j].x, velocities[j].y);
        }
    }
    return 0;
}

