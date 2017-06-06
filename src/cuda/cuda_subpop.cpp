#include "cuda_subpop.h"

void CudaSubpop::evolve(MDVRPProblem* problem, StrIndividual *mngPop, StrIndividual *mngPopRes, int depot, int inds, int lambda) {
    mutate(problem, mngPop, mngPopRes, depot, inds, lambda);
}

void CudaSubpop::mutate(MDVRPProblem* problem, StrIndividual *mngPop, StrIndividual *mngPopRes, int depot, int inds, int lambda) {

    // Call GPU
    cudaMutate(problem, mngPop, mngPopRes, depot, inds, lambda);

}

//void CudaSubpop::processTest(Subpopulation &subpop, IndividualsGroup &offsprings) {
//
//    StrIndividual *mngPop;
//    cudaMallocManaged(&mngPop, subpop.getConfig()->getNumSubIndDepots() * sizeof(StrIndividual));
//    //mngPop.allocate(subpop.getConfig()->getNumSubIndDepots());
//
//    cout << "Time: " << std::time(0) << endl;
//
//    for (int i = 0; i < subpop.getConfig()->getNumSubIndDepots(); ++i) {
//
//        mngPop[i].allocate(subpop.getIndividualsGroup().getIndividuals().at(i).getGene().size());
//
//        for (int j = 0; j < mngPop[i].length; ++j) {
//            mngPop[i].gene[j] = subpop.getIndividualsGroup().getIndividuals().at(i).getGene().at(j);
//        }
//
//        mngPop[i].operations = subpop.getIndividualsGroup().getIndividuals().at(i).autoUpdate(false);
//
//    }
//
//    for (int i = 0; i < subpop.getConfig()->getNumSubIndDepots(); ++i) {
//        cout << i << "-> ";
//        for (int j = 0; j < mngPop[i].length; ++j)
//            cout << mngPop[i].gene[j] << "\t";
//        cout << endl;
//    }
//
//    cout << endl << endl;
//
//    cudaMutate(mngPop, subpop.getConfig()->getNumSubIndDepots());
//
//    for (int i = 0; i < subpop.getConfig()->getNumSubIndDepots(); ++i) {
//        cout << i << "-> ";
//        for (int j = 0; j < mngPop[i].length; ++j)
//            cout << mngPop[i].gene[j] << "\t";
//        cout << endl;
//
//        mngPop[i].deallocate();
//    }
//
//    cout << endl << endl;
//    cudaFree(mngPop);
//
//}