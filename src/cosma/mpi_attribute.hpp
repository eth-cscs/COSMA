namespace cosma {

// An attribute associated with MPI_COMM_SELF.
// MPI_COMM_SELF is guaranteed to be the first object that gets destructed when
// MPI_Finalize is invoked, and is often used in combination with attributes to achieve
// "attribute caching" that substitutes the missing function: MPI_At_finalize.
class mpi_attribute {
public:
    int keyval;

    mpi_attribute();
    ~mpi_attribute();
    void update_attribute(void* value);

private:
    int modify = 1;
};

int delete_fn(MPI_Datatype datatype, int key, void* attr_val, void * extra_state);

}
