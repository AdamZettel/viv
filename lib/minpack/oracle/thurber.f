C SPDX-License-Identifier: MIT OR Apache-2.0
C SPDX-FileCopyrightText: 2026 The Viv authors
C     The oracle for the post: LMDER1 on NIST's Thurber data
C     (lib/minpack/data/Thurber.dat, read from unit 5 starting at its
C     line 61), from NIST's two starting points, written in the format of
C     driver.f. The model is NIST's,
C
C       y = (b1 + b2 x + b3 x^2 + b4 x^3) / (1 + b5 x + b6 x^2 + b7 x^3),
C
C     with residual f_i = model(x_i) - y_i, and the powers of x formed as
C     x2 = x*x and x3 = x2*x, the same operations the Viv post does.
      INTEGER I,INFO,K,LDFJAC,LWA,M,N
      INTEGER IWA(7)
      DOUBLE PRECISION TOL,FNORM2
      DOUBLE PRECISION FJAC(37,7),FVEC(37),WA(5*7+37),X(7),S(7,2)
      DOUBLE PRECISION DPMPAR,ENORM
      DOUBLE PRECISION XD(37),YD(37)
      COMMON /THDATA/ XD,YD
      INTEGER NFEV,NJEV
      COMMON /THCNT/ NFEV,NJEV
      EXTERNAL THFCN
      DATA S /1.0D3,1.0D3,4.0D2,4.0D1,0.7D0,0.3D0,0.03D0,
     *        1.3D3,1.5D3,5.0D2,7.5D1,1.0D0,0.4D0,0.05D0/
      M = 37
      N = 7
      LDFJAC = 37
      LWA = 5*7+37
      TOL = DSQRT(DPMPAR(1))
      DO 5 I = 1, 60
         READ (5,*)
    5    CONTINUE
      DO 10 I = 1, M
         READ (5,*) YD(I),XD(I)
   10    CONTINUE
      OPEN (91, FILE='/dev/stdout', RECL=4000)
      DO 30 K = 1, 2
         DO 20 I = 1, N
            X(I) = S(I,K)
   20       CONTINUE
         WRITE (91,60) 99,N,M,DBLE(K)
         NFEV = 0
         NJEV = 0
         CALL LMDER1(THFCN,M,N,X,FVEC,FJAC,LDFJAC,TOL,INFO,IWA,WA,LWA)
         FNORM2 = ENORM(M,FVEC)
         WRITE (91,70) INFO,NFEV,NJEV,FNORM2,(X(I), I = 1, N)
   30    CONTINUE
      STOP
   60 FORMAT ('C ',3I5,1X,ES25.17E3)
   70 FORMAT ('F ',3I6,99(1X,ES25.17E3))
      END
      SUBROUTINE THFCN(M,N,B,FVEC,FJAC,LDFJAC,IFLAG)
      INTEGER M,N,LDFJAC,IFLAG,I
      DOUBLE PRECISION B(N),FVEC(M),FJAC(LDFJAC,N)
      DOUBLE PRECISION XD(37),YD(37),X,X2,X3,NUM,DEN,T
      COMMON /THDATA/ XD,YD
      INTEGER NFEV,NJEV
      COMMON /THCNT/ NFEV,NJEV
      IF (IFLAG .EQ. 1) NFEV = NFEV + 1
      IF (IFLAG .EQ. 2) NJEV = NJEV + 1
      DO 10 I = 1, M
         X = XD(I)
         X2 = X*X
         X3 = X2*X
         NUM = B(1) + B(2)*X + B(3)*X2 + B(4)*X3
         DEN = 1.0D0 + B(5)*X + B(6)*X2 + B(7)*X3
         IF (IFLAG .EQ. 1) FVEC(I) = NUM/DEN - YD(I)
         IF (IFLAG .EQ. 2) THEN
            FJAC(I,1) = 1.0D0/DEN
            FJAC(I,2) = X/DEN
            FJAC(I,3) = X2/DEN
            FJAC(I,4) = X3/DEN
            T = NUM/(DEN*DEN)
            FJAC(I,5) = -(X*T)
            FJAC(I,6) = -(X2*T)
            FJAC(I,7) = -(X3*T)
         END IF
   10    CONTINUE
      RETURN
      END
