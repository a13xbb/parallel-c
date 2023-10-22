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
int NUM_THREADS = 3;
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

void computeAccelerations() {
    vector **fs;
    fs = malloc(NUM_THREADS * sizeof(vector*));
    for (int i = 0; i < NUM_THREADS; i++) {
        fs[i] = malloc(bodies * sizeof(vector));
        for (int j = 0; j < bodies; j++) {
            fs[i][j].x = 0;
            fs[i][j].y = 0;
        }
    }

    #pragma omp parallel num_threads(NUM_THREADS) 
    {
        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < bodies; i++) {
            int my_rank = omp_get_thread_num();
            // printf("Thread num: %d\n", my_rank);
            for (int j = 0; j < i; j++) {
                vector dist = subtractVectors(positions[i], positions[j]);
                if (dist.x < 1) { //добавляем точкам радиус 0.5
                    dist.x = 1;
                }
                if (dist.y < 1) {
                    dist.y = 1;
                }
                vector dist_neg = scaleVector(-1, dist);

                vector acc_delta = scaleVector(GravConstant / pow(mod(dist), 3), dist_neg);
                fs[my_rank][i] = addVectors(fs[my_rank][i], scaleVector(masses[j], acc_delta));
                fs[my_rank][j] = addVectors(fs[my_rank][j], scaleVector(-masses[i], acc_delta)); 
            }
        }
    }
        
    for (int i = 0; i < bodies; i++) {
        accelerations[i].x = 0;
        accelerations[i].y = 0;
        for (int j = 0; j < NUM_THREADS; j++) {
            accelerations[i] = addVectors(accelerations[i], fs[j][i]);
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
        double start_time = omp_get_wtime();
        for (i = 0; i < timeSteps; i++)
        {
            printf("\nCycle %d\n", i + 1);
            simulate();
            for (j = 0; j < bodies; j++)
                printf("Body %d : %lf\t%lf\t%lf\t%lf\n", j + 1, positions[j].x, positions[j].y, velocities[j].x, velocities[j].y);
        }
        double end_time = omp_get_wtime();
        printf("%.6lf", end_time - start_time);
    }

    return 0;
}

