NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_60
CXXFLAGS = -O3

# Main executable target - compile directly from .cpp
nbody: nbody.cpp
	$(NVCC) $(NVCCFLAGS) -x cu nbody.cpp -o nbody

# Run specific simulation with requested parameters
simulation.out: nbody
	@echo "Starting n-body simulation..."
	@date
	./nbody 100000 0.01 50 10 > simulation.out
	@date
	@echo "Simulation complete."

# Clean up compiled files and outputs
clean:
	rm -f nbody nbody.cu *.out

.PHONY: clean