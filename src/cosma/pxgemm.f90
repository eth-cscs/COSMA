module cosma

  use, intrinsic :: ISO_C_BINDING

  interface
    subroutine psgemm(trans_a, trans_b, m, n, k, &
                    alpha, a, ia, ja, desca,     &
                    b, ib, jb, descb,            &
                    beta, c, ic, jc, descc)      &
    bind(C, name="psgemm")
      use, intrinsic :: ISO_C_BINDING
      implicit none
      character(c_char), intent(in)      :: trans_a,&
                                            trans_b
      integer(c_int), intent(in)         :: m, n, k,&
                                            ia, ja, &
                                            ib, jb, &
                                            ic, jc
      real(c_float), intent(in)          :: alpha,  &
                                            beta
      real(c_float), dimension(*), intent(in) :: a, b
      integer(c_int), dimension(9), intent(in):: desca,&
                                                 descb,&
                                                 descc
      real(c_float), dimension(*), &
                     intent(inout)       :: c
    end subroutine

    subroutine pdgemm(trans_a, trans_b, m, n, k, &
                      alpha, a, ia, ja, desca,   &
                      b, ib, jb, descb,          &
                      beta, c, ic, jc, descc)    &
    bind(C, name="pdgemm")
      use, intrinsic :: ISO_C_BINDING
      implicit none
      character(c_char), intent(in)       :: trans_a,&
                                             trans_b
      integer(c_int), intent(in)          :: m, n, k,&
                                             ia, ja, &
                                             ib, jb, &
                                             ic, jc
      real(c_double), intent(in)          :: alpha, beta
      real(c_double), dimension(*), intent(in) :: a, b
      integer(c_int), dimension(9), intent(in) :: desca,&
                                                  descb,&
                                                  descc
      real(c_double), dimension(*), intent(inout) :: c
    end subroutine

    subroutine pcgemm(trans_a, trans_b, m, n, k, &
                      alpha, a, ia, ja, desca,   &
                      b, ib, jb, descb,          &
                      beta, c, ic, jc, descc)    &
    bind(C, name="pcgemm")
      use, intrinsic :: ISO_C_BINDING
      implicit none
      character(c_char), intent(in)       :: trans_a,&
                                             trans_b
      integer(c_int), intent(in)          :: m, n, k,&
                                             ia, ja, &
                                             ib, jb, &
                                             ic, jc
      complex(c_float_complex), intent(in) :: alpha, &
                                              beta
      complex(c_float_complex), dimension(*), intent(in) :: a, b
      integer(c_int), dimension(9), intent(in) :: desca,&
                                                  descb,&
                                                  descc
      complex(c_float_complex), dimension(*), intent(inout) :: c
    end subroutine

    subroutine pzgemm(trans_a, trans_b, m, n, k, &
                      alpha, a, ia, ja, desca,   &
                      b, ib, jb, descb,          &
                      beta, c, ic, jc, descc)    &
    bind(C, name="pzgemm")
      use, intrinsic :: ISO_C_BINDING
      implicit none
      character(c_char), intent(in)       :: trans_a,&
                                             trans_b
      integer(c_int), intent(in)          :: m, n, k,&
                                             ia, ja, &
                                             ib, jb, &
                                             ic, jc
      complex(c_double_complex), intent(in)          :: alpha,&
                                                        beta
      complex(c_double_complex), dimension(*), intent(in) :: a, b
      integer(c_int), dimension(9), intent(in) :: desca,&
                                                  descb,&
                                                  descc
      complex(c_double_complex), dimension(*), intent(inout) :: c
    end subroutine
  end interface 
end module
