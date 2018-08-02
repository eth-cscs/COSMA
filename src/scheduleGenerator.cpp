#include "scheduleGenerator.h"
#include <tuple>

namespace spartition {

	std::vector<unsigned> Factorize(unsigned input) {
		std::vector<unsigned> factors;
		unsigned i = 2;
		while (i <= sqrt(input)) {
			if (input % i == 0) {
				factors.push_back(i);
				input /= i;
			}
			else {
				i++;
			}
		}
		if (input > 1) {
			factors.push_back(input);
		}
		return factors;
	}

	// find all divisors of a given number n
	std::vector<int> find_divisors(int n) {
		std::vector<int> divs;
		for (int i = 2; i < n; ++i) {
			if (n % i == 0) {
				divs.push_back(i);
			}
		}
		return divs;
	}

	int closest_divisor(int P, int dimension, double target) {
		int divisor = 1;
		int error;
		int best_error = std::numeric_limits<int>::max();
		int best_div = 1;

		for (int i : find_divisors(P)) {
			error = std::abs(1.0*dimension / i - target);

			if (error < best_error) {
				best_div = i;
				best_error = error;
			}
		}

		return best_div;
	}

	std::tuple<int, int, int> balanced_divisors(long long m, long long n, long long k, int P) {
		// sort the dimensions 
		std::vector<long long> dimensions = { m, n, k };
		std::sort(dimensions.begin(), dimensions.end());

		// find divm, divn, divk such that m/divm = n/divn = k/divk (as close as possible)
		// be careful when dividing, since the product mnk can be very large
		double target_tile_size = std::cbrt(1.0*dimensions[1] * dimensions[2] / P * dimensions[0]);
		int divk = closest_divisor(P, k, target_tile_size);
		P /= divk;
		int divn = closest_divisor(P, n, target_tile_size);
		P /= divn;
		int divm = P;

		return std::make_tuple(divm, divn, divk);
	}

	Schedule GetDFSSchedule(unsigned localM, unsigned localN, unsigned localK, unsigned S, DFSSchedule DFSsched) {
		unsigned a, b, c;
		if (DFSsched == DFSSchedule::DFScubic) {
			unsigned cubicSide = std::floor(std::sqrt(S / 3.0));
			a = std::min(cubicSide, localM);
			b = std::min(cubicSide, localN);
			c = std::min(cubicSide, localK);
		}
		else if (DFSsched == DFSSchedule::DFSsquare) {
			unsigned squareSide = std::floor(std::sqrt(S + 1.0) - 1);
			a = std::min(squareSide, localM);
			b = std::min(squareSide, localN);
			c = std::min(std::max((unsigned)1, (unsigned)std::floor((S - a * b) / (a + b))), localK);
		}
		Schedule sched;
		sched.tileSizeK = c;
		sched.tileSizeM = a;
		sched.tileSizeN = b;
		//update. We only do DFS in K dimension
		sched.divisions = std::vector<SingleStep>(1);
		sched.divisions[0].Dim = dim::dimK;
		sched.divisions[0].SplitType = splitType::DFS;
		sched.divisions[0].SplitSize = (localK - 1) / c + 1;

		//sched.divisions = std::vector<SingleStep>(3);
		//sched.divisions[0].Dim = dim::dimM;
		//sched.divisions[0].SplitType = splitType::DFS;
		//sched.divisions[0].SplitSize = (localM - 1) / a + 1;

		//sched.divisions[1].Dim = dim::dimN;
		//sched.divisions[1].SplitType = splitType::DFS;
		//sched.divisions[1].SplitSize = (localN - 1) / b + 1;

		//sched.divisions[2].Dim = dim::dimK;
		//sched.divisions[2].SplitType = splitType::DFS;
		//sched.divisions[2].SplitSize = (localK - 1) / c + 1;
		//#if (DFS_SCHEDULE == NATIVE_DFS)
		//#endif
		return sched;
	}

	std::vector<unsigned> Intersect(std::vector<unsigned> v1, std::vector<unsigned> v2) {
		std::vector<unsigned> v3;
		sort(v1.begin(), v1.end());
		sort(v2.begin(), v2.end());

		set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(v3));

		return v3;
	}

	double EvaluateCommCost(long long M, long long N, long long K, int gridM, int gridN, int gridK) {
		int sizeM = M / gridM;
		int sizeN = N / gridN;
		int sizeK = K / gridK;
		long long perTileCost = sizeM * sizeN + sizeM * sizeK + sizeN * sizeK;
		double total_cost = 1.0 * perTileCost * gridM * gridN * gridK;
		return total_cost;
	}

	std::tuple<unsigned, unsigned, unsigned> FindGrid(unsigned p, long long M, long long N, long long K,
		std::tuple<unsigned, unsigned, unsigned> grid,
		double maxCompLoss = 0.03, double maxCommLoss = 0.2) {
		unsigned gridM, gridN, gridK;
		std::tie(gridM, gridN, gridK) = grid;
		unsigned lostProcesses = p - gridM * gridN * gridK;

		//if we loose too many processes, try to find something better
		double p_ratio = 1.0 * lostProcesses / p;
		if (p_ratio > maxCompLoss) {
			double optCommCost = EvaluateCommCost(M, N, K, gridM, gridN, gridK);
			double curCommCost = std::numeric_limits<double>::max();
			unsigned gridMcurrent, gridNcurrent, gridKcurrent;
			for (unsigned i = 0; i < p * maxCompLoss; i++) {
				if (1.0 * curCommCost / optCommCost > 1 + maxCommLoss) {
					std::tie(gridMcurrent, gridNcurrent, gridKcurrent) = balanced_divisors(M, N, K, p - i);
					curCommCost = EvaluateCommCost(M, N, K, gridMcurrent, gridNcurrent, gridKcurrent);
				}
				else {
					gridM = gridMcurrent;
					gridN = gridNcurrent;
					gridK = gridKcurrent;
					break;
				}
			}
		}
		return std::make_tuple(gridM, gridN, gridK);
	}

	Schedule GenerateSchedule(ProblemParameters params) {
		Schedule sched;
		switch (params.schedule) {
			case (schedType::S2D): {
				int squaretileSize = (int)std::sqrt(1LL * params.m*params.n / params.P);
				sched.numTilesM = (params.m - 1) / squaretileSize + 1;
				sched.numTilesN = (params.n - 1) / squaretileSize + 1;
				while (sched.numTilesM * sched.numTilesN > params.P) {
					int newTileSizeM = sched.numTilesM == 1 ? MAX_SIZE : (params.m - 1) / (sched.numTilesM - 1) + 1;
					int newTileSizeN = sched.numTilesN == 1 ? MAX_SIZE : (params.n - 1) / (sched.numTilesN - 1) + 1;
					squaretileSize = std::min(newTileSizeM, newTileSizeN);
					sched.numTilesM = (params.m - 1) / squaretileSize + 1;
					sched.numTilesN = (params.n - 1) / squaretileSize + 1;
				}
				sched.tileSizeM = (params.m - 1) / sched.numTilesM + 1;
				sched.tileSizeN = (params.n - 1) / sched.numTilesN + 1;
				sched.tileSizeK = params.k;
				sched.numTilesK = 1;
				break;
			}
			case (schedType::S3D): {
				//TODO : out of range problems
				int cubicTileSize = (int)std::cbrt(1LL * params.k * params.m * params.n / params.P);

				sched.numTilesM = (params.m - 1) / cubicTileSize + 1;
				sched.numTilesN = (params.n - 1) / cubicTileSize + 1;
				sched.numTilesK = (params.k - 1) / cubicTileSize + 1;
				while (sched.numTilesM * sched.numTilesN * sched.numTilesK > params.P)
				{
					unsigned newTileSizeM = sched.numTilesM == 1 ? MAX_SIZE : (params.m - 1) / (sched.numTilesM - 1) + 1;
					unsigned newTileSizeN = sched.numTilesN == 1 ? MAX_SIZE : (params.n - 1) / (sched.numTilesN - 1) + 1;
					unsigned newTileSizeK = sched.numTilesK == 1 ? MAX_SIZE : (params.k - 1) / (sched.numTilesK - 1) + 1;
					cubicTileSize = std::min(std::min(newTileSizeM, newTileSizeN), newTileSizeK);
					sched.numTilesM = (params.m - 1) / cubicTileSize + 1;
					sched.numTilesN = (params.n - 1) / cubicTileSize + 1;
					sched.numTilesK = (params.k - 1) / cubicTileSize + 1;
				}
				break;
			}
			case (schedType::COMM): {
				//TODO : out of range problems
				int a = (int)std::min(std::sqrt(params.S), std::cbrt(1LL * params.k * params.m * params.n / params.P));
				int b = (int)std::max((double)(params.m * params.n * params.k) / (params.P * params.S), std::cbrt(1LL * params.k * params.m * params.n / params.P));

				sched.numTilesM = (params.m - 1) / a + 1;
				sched.numTilesN = (params.n - 1) / a + 1;
				sched.numTilesK = (params.k - 1) / b + 1;

				sched.tileSizeM = (params.m - 1) / sched.numTilesM + 1;
				sched.tileSizeN = (params.n - 1) / sched.numTilesN + 1;
				sched.tileSizeK = (params.k - 1) / sched.numTilesK + 1;
				bool good = true;

				//std::cout << "Proc grid: [" << sched.numTilesM << " x " << sched.numTilesN << " x " << sched.numTilesK << "]\n";
				while (sched.numTilesM * sched.numTilesN * sched.numTilesK > params.P)
				{
					good = false;
					if (a < std::sqrt(params.S)) {
						//find which dimension requires least stretching 
						unsigned newTileSizeM = sched.numTilesM == 1 ? MAX_SIZE : (params.m - 1) / (sched.numTilesM - 1) + 1;
						unsigned newTileSizeN = sched.numTilesN == 1 ? MAX_SIZE : (params.n - 1) / (sched.numTilesN - 1) + 1;
						unsigned newTileSizeK = sched.numTilesK == 1 ? MAX_SIZE : (params.k - 1) / (sched.numTilesK - 1) + 1;
						if ((newTileSizeK <= newTileSizeM)  && (newTileSizeK <= newTileSizeN)) {
							sched.numTilesK = (params.k - 1) / newTileSizeK + 1;
						}
						else {
							if (newTileSizeN < newTileSizeM && newTileSizeN * sched.tileSizeM < params.S) {
								sched.numTilesN = (params.n - 1) / newTileSizeN + 1;
							}
							else if (newTileSizeM * sched.tileSizeN < params.S) {
								sched.numTilesM = (params.m - 1) / newTileSizeM + 1;
							}
							else {
								sched.numTilesK = (params.k - 1) / newTileSizeK + 1;
							}
						}
						if (sched.numTilesM * sched.numTilesN * sched.numTilesK <= params.P) {
							good = true;
							break;
						}
					}
					else {
						sched.numTilesK = params.P / (sched.numTilesM *sched.numTilesN);
						if (sched.numTilesM * sched.numTilesN * sched.numTilesK <= params.P) {
							good = true;
							break;
						}
					}
					//std::cout << "Proc grid: [" << sched.numTilesM << " x " << sched.numTilesN << " x " << sched.numTilesK << "]\n";
				}
				break;
			}
		}

		//physical num cores refinement
		std::tie(sched.numTilesM, sched.numTilesN, sched.numTilesK) =
			FindGrid(params.P, params.m, params.n, params.k, std::make_tuple(sched.numTilesM, sched.numTilesN, sched.numTilesK));
		// sched.numTilesM = FitRanks(params.numCores, sched.numTilesM);
		// sched.numTilesN = FitRanks(params.numCores, sched.numTilesN);
		// sched.numTilesK = FitRanks(params.numCores, sched.numTilesK);

		sched.tileSizeM = (params.m - 1) / sched.numTilesM + 1;
		sched.tileSizeN = (params.n - 1) / sched.numTilesN + 1;
		sched.tileSizeK = (params.k - 1) / sched.numTilesK + 1;

		double localMemCapacity = 1.0 * params.S / std::max(sched.tileSizeM, sched.tileSizeN);

		if (params.divStrat == DivisionStrategy::oneStep) {
			sched.divisions = std::vector<SingleStep>(3);
			sched.divisions[0].Dim = dim::dimM;
			sched.divisions[0].SplitSize = sched.numTilesM;
			sched.divisions[0].SplitType = splitType::BFS;

			sched.divisions[1].Dim = dim::dimN;
			sched.divisions[1].SplitSize = sched.numTilesN;
			sched.divisions[1].SplitType = splitType::BFS;

			sched.divisions[2].Dim = dim::dimK;
			sched.divisions[2].SplitSize = sched.numTilesK;
			sched.divisions[2].SplitType = splitType::BFS;
		}

		else {
			std::vector<unsigned> divisionsM = Factorize(sched.numTilesM);
			std::vector<unsigned> divisionsN = Factorize(sched.numTilesN);
			std::vector<unsigned> divisionsK = Factorize(sched.numTilesK);

			//TODO: should we split the DFS steps too?

			//std::vector<unsigned> divisionsDFS = Factorize(numDFSsteps);
			std::vector<unsigned> sortedDivisions = { (unsigned)divisionsM.size(), (unsigned)divisionsN.size(), (unsigned)divisionsK.size() };
			std::sort(sortedDivisions.begin(), sortedDivisions.end());
			sched.divisions = std::vector<SingleStep>(divisionsM.size() + divisionsN.size() + divisionsK.size());

			unsigned counter = 0;
			//TODO: does the order matter?
			for (size_t i = 0; i < sortedDivisions[2]; i++)
			{
				if (divisionsM.size() > i) {
					sched.divisions[counter].Dim = dim::dimM;
					sched.divisions[counter].SplitSize = divisionsM[i];
					sched.divisions[counter].SplitType = splitType::BFS;
					counter++;
				}
				if (divisionsN.size() > i) {
					sched.divisions[counter].Dim = dim::dimN;
					sched.divisions[counter].SplitSize = divisionsN[i];
					sched.divisions[counter].SplitType = splitType::BFS;
					counter++;
				}
				if (divisionsK.size() > i) {
					sched.divisions[counter].Dim = dim::dimK;
					sched.divisions[counter].SplitSize = divisionsK[i];
					sched.divisions[counter].SplitType = splitType::BFS;
					counter++;
				}
			}
		}

		Schedule dfsSched = GetDFSSchedule(sched.tileSizeM, sched.tileSizeN, sched.tileSizeK, params.S, params.dfsSched);
		if (dfsSched.divisions[0].SplitSize > 1) {
			sched.divisions.insert(sched.divisions.end(), dfsSched.divisions.begin(), dfsSched.divisions.end());
			sched.tileSizeM = dfsSched.tileSizeM;
			sched.tileSizeN = dfsSched.tileSizeN;
			sched.tileSizeK = dfsSched.tileSizeK;
		}
		return sched;
	};
}
