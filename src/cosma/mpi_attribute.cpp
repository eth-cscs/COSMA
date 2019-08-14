#include <mpi.h>
#include <cosma/mpi_attribute.hpp>

namespace cosma {
mpi_attribute::mpi_attribute() {
    // create a keyval.
    MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, delete_fn,
        &keyval, (void*) &modify);
}

mpi_attribute::~mpi_attribute() {
    int finalized = 0;
    MPI_Finalized(&finalized);
    // if MPI is already finalized, then MPI_COMM_SELF
    // is already destroyed, together with all its attributes
    // so there is no need to destroy them here.
    if (!finalized) {
        modify = 0;
        MPI_Comm_delete_attr(MPI_COMM_SELF, keyval);
        MPI_Comm_free_keyval(&keyval);
    }
}

// assign the attribute value to the keyval.
// MPI_Comm_set_attr will invoke delete_fn if the attribute already exists. 
// What we want here instead is to just update the value of the attribute, without previously 
// invoking delete_fn, because our delete_fn function will deallocate all MPI buffers. 
// For this reason, we set the flag modify to false, before invoking the set function.
void mpi_attribute::update_attribute(void* value) {
    // MPI_Comm_set_attr invokes delete_fn.
    // set modify to false to prevent this.
    // We want delete_fn to be invoked by MPI_Finalized, not here.
    modify = 0;
    MPI_Comm_set_attr(MPI_COMM_SELF, keyval, value);
    // set modify to true, so that if MPI_Finalize can delete this attribute
    // if it gets invokes before this object (i.e. also the context) gets destroyed.
    modify = 1;
}

// we want to associate this attribute with MPI_COMM_SELF
// so that it is destroyed when MPI_Finalize is invoked
// (in fact, it is the first object to get destroyed in MPI_Finalize).
int delete_fn(MPI_Datatype datatype, int key, void* attr_val, void * extra_state) {
    int* ptr = (int*) attr_val;
    int* modify = (int*) extra_state;
    if ((*modify == 1) && ptr) {
        MPI_Free_mem(ptr);
    }
    return MPI_SUCCESS;
}

}
