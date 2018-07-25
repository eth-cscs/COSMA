#pragma once
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <assert.h>

#define MAX_SIZE 2147483640 

namespace spartition {

enum dim { dimM, dimN, dimK };

enum splitType { BFS, DFS };

enum schedType { S2D, S3D, S25D };

enum DivisionStrategy { oneStep, recursive };

enum DFSSchedule { DFScubic, DFSsquare };

struct SingleStep {
	dim Dim;
	splitType SplitType;
	unsigned SplitSize;
};

struct Schedule {
	std::vector<SingleStep> divisions;
	unsigned tileSizeM, tileSizeN, tileSizeK;
	unsigned numTilesM, numTilesN, numTilesK;

	void Print() {
		std::cout << "\nTile size [" << tileSizeM << " x " << tileSizeN << " x " << tileSizeK << "]. Number of tiles "
			<< numTilesM << " x " << numTilesN << " x " << numTilesK << "\n";
		std::cout << "\nDivision strategy:\n[dim] ; [size] ; [BFS/DFS]\n";
		for (auto step : divisions) {
			if (step.Dim == dimM) std::cout << "m ; ";
			if (step.Dim == dimN) std::cout << "n ; ";
			if (step.Dim == dimK) std::cout << "k ; ";
			std::cout << step.SplitSize << " ; ";
			if (step.SplitType == BFS) std::cout << "BFS\n";
			if (step.SplitType == DFS) std::cout << "DFS\n";
		}
	}
};

struct ProblemParameters {
	long long  m, n, k; //dimensions
	long long  S; //size of local memory
	long long  numCores; //number of cores per node
	long long  P; //number of processes (ranks)
	schedType schedule; //generates 2d or 3d schedule (parallelization in k dimension)
	DivisionStrategy divStrat; //e.g., if schedule should look like 16 or 2, 2, 2, 2
	DFSSchedule dfsSched; //cubic vs square tiling
};

std::vector<unsigned> Factorize(unsigned);

Schedule GenerateSchedule(ProblemParameters params);
}
