#include "scheduleGenerator.h"

namespace spartition {
std::vector<int> Factorize(int input) {
	std::vector<int> factors;
	int i = 2;
    while (i <= std::sqrt(input)){
		if (input % i == 0) {
			factors.push_back(i);
			input /= i;
		}
		else {
			i++;
		}
	}
	if (input > 1) factors.push_back(input);
	return factors;
}

Schedule GenerateSchedule(ProblemParameters params) {
	Schedule sched;
	if (params.schedule == schedType::S2D) {
        int squaretileSize = (int) std::sqrt(1LL * params.m*params.n / params.P);
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
	}
	else {
		//TODO : out of range problems
        int cubicTileSize = (int) std::cbrt(1LL * params.k * params.m * params.n / params.P);
       
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
	}

	//physical num cores refinement
	sched.numTilesM = FitRanks(params.numCores, sched.numTilesM);
	sched.numTilesN = FitRanks(params.numCores, sched.numTilesN);
	sched.numTilesK = FitRanks(params.numCores, sched.numTilesK);
	
	sched.tileSizeM = (params.m - 1) / sched.numTilesM + 1;
	sched.tileSizeN = (params.n - 1) / sched.numTilesN + 1;
	sched.tileSizeK = (params.k - 1) / sched.numTilesK + 1;

	unsigned localMemCapacity = params.S / std::max(sched.tileSizeM, sched.tileSizeN);

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
		std::vector<unsigned> sortedDivisions  =  { (unsigned)divisionsM.size(), (unsigned)divisionsN.size(), (unsigned)divisionsK.size() };
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
	sched.divisions.insert(sched.divisions.end(), dfsSched.divisions.begin(), dfsSched.divisions.end());
	sched.tileSizeM = dfsSched.tileSizeM;
	sched.tileSizeN = dfsSched.tileSizeN;
	sched.tileSizeK = dfsSched.tileSizeK;
		return sched;
}
