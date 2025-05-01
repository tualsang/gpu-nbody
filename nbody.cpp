#include <iostream>
#include <fstream>
#include <random>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

double G = 6.674*std::pow(10,-11);
//double G = 1;

struct simulation {
  size_t nbpart;
  
  std::vector<double> mass;

  //position
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;

  //velocity
  std::vector<double> vx;
  std::vector<double> vy;
  std::vector<double> vz;

  //force
  std::vector<double> fx;
  std::vector<double> fy;
  std::vector<double> fz;

  
  simulation(size_t nb)
    :nbpart(nb), mass(nb),
     x(nb), y(nb), z(nb),
     vx(nb), vy(nb), vz(nb),
     fx(nb), fy(nb), fz(nb) 
  {}
};


void random_init(simulation& s) {
  std::random_device rd;  
  std::mt19937 gen(rd());
  std::uniform_real_distribution dismass(0.9, 1.);
  std::normal_distribution dispos(0., 1.);
  std::normal_distribution disvel(0., 1.);

  for (size_t i = 0; i<s.nbpart; ++i) {
    s.mass[i] = dismass(gen);

    s.x[i] = dispos(gen);
    s.y[i] = dispos(gen);
    s.z[i] = dispos(gen);
    s.z[i] = 0.;
    
    s.vx[i] = disvel(gen);
    s.vy[i] = disvel(gen);
    s.vz[i] = disvel(gen);
    s.vz[i] = 0.;
    s.vx[i] = s.y[i]*1.5;
    s.vy[i] = -s.x[i]*1.5;
  }

  return;
  //normalize velocity (using normalization found on some physicis blog)
  double meanmass = 0;
  double meanmassvx = 0;
  double meanmassvy = 0;
  double meanmassvz = 0;
  for (size_t i = 0; i<s.nbpart; ++i) {
    meanmass += s.mass[i];
    meanmassvx += s.mass[i] * s.vx[i];
    meanmassvy += s.mass[i] * s.vy[i];
    meanmassvz += s.mass[i] * s.vz[i];
  }
  for (size_t i = 0; i<s.nbpart; ++i) {
    s.vx[i] -= meanmassvx/meanmass;
    s.vy[i] -= meanmassvy/meanmass;
    s.vz[i] -= meanmassvz/meanmass;
  }
  
}

void init_solar(simulation& s) {
  enum Planets {SUN, MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE, MOON};
  s = simulation(10);

  // Masses in kg
  s.mass[SUN] = 1.9891 * std::pow(10, 30);
  s.mass[MERCURY] = 3.285 * std::pow(10, 23);
  s.mass[VENUS] = 4.867 * std::pow(10, 24);
  s.mass[EARTH] = 5.972 * std::pow(10, 24);
  s.mass[MARS] = 6.39 * std::pow(10, 23);
  s.mass[JUPITER] = 1.898 * std::pow(10, 27);
  s.mass[SATURN] = 5.683 * std::pow(10, 26);
  s.mass[URANUS] = 8.681 * std::pow(10, 25);
  s.mass[NEPTUNE] = 1.024 * std::pow(10, 26);
  s.mass[MOON] = 7.342 * std::pow(10, 22);

  // Positions (in meters) and velocities (in m/s)
  double AU = 1.496 * std::pow(10, 11); // Astronomical Unit

  s.x = {0, 0.39*AU, 0.72*AU, 1.0*AU, 1.52*AU, 5.20*AU, 9.58*AU, 19.22*AU, 30.05*AU, 1.0*AU + 3.844*std::pow(10, 8)};
  s.y = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  s.z = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  s.vx = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  s.vy = {0, 47870, 35020, 29780, 24130, 13070, 9680, 6800, 5430, 29780 + 1022};
  s.vz = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
}

//meant to update the force that from applies on to
void update_force(simulation& s, size_t from, size_t to) {
  double softening = .1;
  double dist_sq = std::pow(s.x[from]-s.x[to],2)
    + std::pow(s.y[from]-s.y[to],2)
    + std::pow(s.z[from]-s.z[to],2);
  double F = G * s.mass[from]*s.mass[to]/(dist_sq+softening); //that the strength of the force

  //direction
  double dx = s.x[from]-s.x[to];
  double dy = s.y[from]-s.y[to];
  double dz = s.z[from]-s.z[to];
  double norm = std::sqrt(dx*dx+dy*dy+dz*dz);
  
  dx = dx/norm;
  dy = dy/norm;
  dz = dz/norm;

  //apply force
  s.fx[to] += dx*F;
  s.fy[to] += dy*F;
  s.fz[to] += dz*F;
}

void reset_force(simulation& s) {
  for (size_t i=0; i<s.nbpart; ++i) {
    s.fx[i] = 0.;
    s.fy[i] = 0.;
    s.fz[i] = 0.;
  }
}

/**
 * Kernel to calculate gravitational forces between all particles
 * Each thread handles one particle
 */
__global__ void updateForceKernel(double* mass, double* x, double* y, double* z, double* fx, double* fy, double* fz, size_t nbpart, double G, double softening) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nbpart) return; // Check bounds
  
  // Reset forces for this particle
  fx[idx] = 0.0;
  fy[idx] = 0.0;
  fz[idx] = 0.0;
  
  double dx, dy, dz, dist_sq, F, norm;
  for (size_t j = 0; j < nbpart; ++j) {
    if (idx != j) {  // Skip self-interaction
      dx = x[j] - x[idx];
      dy = y[j] - y[idx];
      dz = z[j] - z[idx];
      
      dist_sq = dx*dx + dy*dy + dz*dz + softening;
      F = G * mass[idx] * mass[j] / dist_sq;
      
      norm = sqrt(dist_sq);
      dx /= norm;
      dy /= norm;
      dz /= norm;
      
      // Apply force
      fx[idx] += dx * F;
      fy[idx] += dy * F;
      fz[idx] += dz * F;
    }
  }
}

/**
 * Kernel to update positions and velocities of all particles
 * Each thread handles one particle
 */
__global__ void updatePositionVelocityKernel(double* mass, double* x, double* y, double* z, 
  double* vx, double* vy, double* vz,
  double* fx, double* fy, double* fz, 
  size_t nbpart, double dt) {
// Get global thread ID
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= nbpart) return; // Check bounds

// Cache this particle's mass and current position/velocity
double m = mass[idx];
double pos_x = x[idx];
double pos_y = y[idx];
double pos_z = z[idx];
double vel_x = vx[idx];
double vel_y = vy[idx];
double vel_z = vz[idx];

// Get forces
double f_x = fx[idx];
double f_y = fy[idx];
double f_z = fz[idx];

// Update velocity: v = v + F/m * dt (a = F/m)
vel_x += f_x / m * dt;
vel_y += f_y / m * dt;
vel_z += f_z / m * dt;

// Update position: p = p + v * dt
pos_x += vel_x * dt;
pos_y += vel_y * dt;
pos_z += vel_z * dt;

// Write updated values back to global memory
vx[idx] = vel_x;
vy[idx] = vel_y;
vz[idx] = vel_z;
x[idx] = pos_x;
y[idx] = pos_y;
z[idx] = pos_z;
}

void dump_state(simulation& s) {
  std::cout<<s.nbpart<<'\t';
  for (size_t i=0; i<s.nbpart; ++i) {
    std::cout<<s.mass[i]<<'\t';
    std::cout<<s.x[i]<<'\t'<<s.y[i]<<'\t'<<s.z[i]<<'\t';
    std::cout<<s.vx[i]<<'\t'<<s.vy[i]<<'\t'<<s.vz[i]<<'\t';
    std::cout<<s.fx[i]<<'\t'<<s.fy[i]<<'\t'<<s.fz[i]<<'\t';
  }
  std::cout<<'\n';
}

void load_from_file(simulation& s, std::string filename) {
  std::ifstream in (filename);
  size_t nbpart;
  in>>nbpart;
  s = simulation(nbpart);
  for (size_t i=0; i<s.nbpart; ++i) {
    in>>s.mass[i];
    in >>  s.x[i] >>  s.y[i] >>  s.z[i];
    in >> s.vx[i] >> s.vy[i] >> s.vz[i];
    in >> s.fx[i] >> s.fy[i] >> s.fz[i];
  }
  if (!in.good())
    throw "kaboom";
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    std::cerr
      <<"usage: "<<argv[0]<<" <input> <dt> <nbstep> <printevery>"<<"\n"
      <<"input can be:"<<"\n"
      <<"a number (random initialization)"<<"\n"
      <<"planet (initialize with solar system)"<<"\n"
      <<"a filename (load from file in singleline tsv)"<<"\n";
    return -1;
  }
  
  double dt = std::atof(argv[2]); //in seconds
  size_t nbstep = std::atol(argv[3]);
  size_t printevery = std::atol(argv[4]);
  
  simulation s(1);

  //parse command line
  {
    size_t nbpart = std::atol(argv[1]); //return 0 if not a number
    if ( nbpart > 0) {
      s = simulation(nbpart);
      random_init(s);
    } else {
      std::string inputparam = argv[1];
      if (inputparam == "planet") {
        init_solar(s);
      } else{
        load_from_file(s, inputparam);
      }
    }    
  }

  // Allocate memory on GPU
  double *d_mass, *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_fx, *d_fy, *d_fz;
  cudaMalloc((void**)&d_mass, s.nbpart * sizeof(double));
  cudaMalloc((void**)&d_x, s.nbpart * sizeof(double));
  cudaMalloc((void**)&d_y, s.nbpart * sizeof(double));
  cudaMalloc((void**)&d_z, s.nbpart * sizeof(double));
  cudaMalloc((void**)&d_vx, s.nbpart * sizeof(double));
  cudaMalloc((void**)&d_vy, s.nbpart * sizeof(double));
  cudaMalloc((void**)&d_vz, s.nbpart * sizeof(double));
  cudaMalloc((void**)&d_fx, s.nbpart * sizeof(double));
  cudaMalloc((void**)&d_fy, s.nbpart * sizeof(double));
  cudaMalloc((void**)&d_fz, s.nbpart * sizeof(double));

  // Copy initial data from host to device
  cudaMemcpy(d_mass, s.mass.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, s.x.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, s.y.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, s.z.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vx, s.vx.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vy, s.vy.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vz, s.vz.data(), s.nbpart * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemset(d_fx, 0, s.nbpart * sizeof(double));
  cudaMemset(d_fy, 0, s.nbpart * sizeof(double));
  cudaMemset(d_fz, 0, s.nbpart * sizeof(double));

  // Configure kernel launch parameters
  int threadsPerBlock = BLOCK_SIZE;
  int numBlocks = (s.nbpart + threadsPerBlock - 1) / threadsPerBlock;
  double softening = 0.1;
  
  // For optimal GPU utilization
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  printf("CUDA Device: %s\n", deviceProp.name);
  printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
  printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
  printf("Multiprocessor count: %d\n", deviceProp.multiProcessorCount);

  // Main simulation loop
  printf("Starting simulation with %zu particles for %zu steps...\n", s.nbpart, nbstep);
  printf("Output will be printed every %zu steps\n", printevery);
  
  // Print initial state
  printf("Initial state (step 0):\n");
  dump_state(s);
  
  // Record timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  
  // Main time-stepping loop
  for (size_t step = 1; step <= nbstep; step++) {
    //Calculate gravitational forces between all particles
    updateForceKernel<<<numBlocks, threadsPerBlock>>>(
      d_mass, d_x, d_y, d_z, d_fx, d_fy, d_fz, s.nbpart, G, softening);
    
    //Update velocities and positions based on calculated forces
    updatePositionVelocityKernel<<<numBlocks, threadsPerBlock>>>(
      d_mass, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, s.nbpart, dt);
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA Error during step %zu: %s\n", step, cudaGetErrorString(err));
      break;
    }
    
    //At regular intervals, copy data back and print state
    if (step % printevery == 0) {
      // Copy all data back from device to host for output
      cudaMemcpy(s.x.data(), d_x, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(s.y.data(), d_y, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(s.z.data(), d_z, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(s.vx.data(), d_vx, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(s.vy.data(), d_vy, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(s.vz.data(), d_vz, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(s.fx.data(), d_fx, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(s.fy.data(), d_fy, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(s.fz.data(), d_fz, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
      
      // Print progress and state
      printf("Step %zu of %zu (%.1f%%):\n", step, nbstep, step * 100.0 / nbstep);
      dump_state(s);
    }
  }
  
  // Record end time and calculate total time
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Simulation completed in %.2f milliseconds\n", elapsedTime);
  
  // If the last step wasn't printed, copy back and print final state
  if (nbstep % printevery != 0) {
    // Copy all data back from device to host
    cudaMemcpy(s.x.data(), d_x, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(s.y.data(), d_y, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(s.z.data(), d_z, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(s.vx.data(), d_vx, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(s.vy.data(), d_vy, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(s.vz.data(), d_vz, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(s.fx.data(), d_fx, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(s.fy.data(), d_fy, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(s.fz.data(), d_fz, s.nbpart * sizeof(double), cudaMemcpyDeviceToHost);
    
    printf("Final state (step %zu):\n", nbstep);
    dump_state(s);
  }

  // Free device memory
  cudaFree(d_mass);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_vx);
  cudaFree(d_vy);
  cudaFree(d_vz);
  cudaFree(d_fx);
  cudaFree(d_fy);
  cudaFree(d_fz);

  return 0;
}