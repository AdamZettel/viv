! SPDX-License-Identifier: CC0-1.0
! SPDX-FileCopyrightText: 2026 The Viv authors
! The uniform random numbers of the comparison builds of upstream: Viv's
! rand(), Park-Miller MINSTD (state' = 16807 state mod (2^31 - 1), a draw is
! state / (2^31 - 1)), seeded as Viv's srand(seed) seeds it. patch.py
! replaces upstream's CALL RANDOM_INIT by CALL minstd_seed and its CALL
! RANDOM_NUMBER by CALL minstd_number, so a patched upstream program and
! the Viv port draw the same numbers in the same order. The seed is read
! from the environment variable MINSTD_SEED (default 1).
MODULE minstd_module
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: minstd_seed, minstd_number
  INTEGER(8), SAVE :: state = 1_8
  INTEGER(8), PARAMETER :: m = 2147483647_8, a = 16807_8
  INTERFACE minstd_number
     MODULE PROCEDURE number_0, number_1, number_2
  END INTERFACE minstd_number
CONTAINS
  SUBROUTINE minstd_seed ( repeatable, image_distinct )
    LOGICAL, INTENT(in) :: repeatable, image_distinct
    CHARACTER(len=32) :: text
    INTEGER           :: length, status
    REAL(8)           :: seed
    seed = 1.0d0
    CALL GET_ENVIRONMENT_VARIABLE ( 'MINSTD_SEED', text, length, status )
    IF ( status == 0 .AND. length > 0 ) READ ( text, * ) seed
    state = MOD ( INT ( FLOOR ( ABS ( seed ) ), 8 ), m - 1_8 ) + 1_8
    IF ( repeatable .OR. image_distinct ) CONTINUE
  END SUBROUTINE minstd_seed
  FUNCTION draw () RESULT ( x )
    REAL(8) :: x
    state = MOD ( a * state, m )
    x = REAL ( state, 8 ) / REAL ( m, 8 )
  END FUNCTION draw
  SUBROUTINE number_0 ( x )
    REAL(8), INTENT(out) :: x
    x = draw()
  END SUBROUTINE number_0
  SUBROUTINE number_1 ( x )
    REAL(8), DIMENSION(:), INTENT(out) :: x
    INTEGER :: i
    DO i = 1, SIZE(x)
       x(i) = draw()
    END DO
  END SUBROUTINE number_1
  SUBROUTINE number_2 ( x )
    REAL(8), DIMENSION(:,:), INTENT(out) :: x
    INTEGER :: i, j
    DO j = 1, SIZE(x,2)
       DO i = 1, SIZE(x,1)
          x(i,j) = draw()
       END DO
    END DO
  END SUBROUTINE number_2
END MODULE minstd_module
