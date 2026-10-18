C SPDX-License-Identifier: MIT OR Apache-2.0
C SPDX-FileCopyrightText: 2026 The Viv authors
C     The oracle driver: runs LMDER1 on the cases of the netlib test data
C     (ref/file22.dat, the input of ref/file17.f) and writes every number at
C     full precision, one case after another:
C
C       C  nprob n m factor
C       S  nfev fnorm x(1..n)                     (traced build only)
C       T  one line per trial step                (traced build only)
C       F  info nfev njev fnorm x(1..n)
C
C     It uses the problem routines FCN, SSQFCN, SSQJAC and INITPT of
C     ref/file17.f unchanged (oracle/build.sh cuts them out of that file).
      INTEGER I,INFO,K,LDFJAC,LWA,M,N,NFEV,NJEV,NPROB,NTRIES
      INTEGER IWA(40)
      DOUBLE PRECISION FACTOR,FNORM2,TOL
      DOUBLE PRECISION FJAC(65,40),FVEC(65),WA(265),X(40)
      DOUBLE PRECISION DPMPAR,ENORM
      EXTERNAL FCN
      COMMON /REFNUM/ NPROB,NFEV,NJEV
      TOL = DSQRT(DPMPAR(1))
      LDFJAC = 65
      LWA = 265
      OPEN (91, FILE='/dev/stdout', RECL=4000)
   10 CONTINUE
         READ (5,50) NPROB,N,M,NTRIES
         IF (NPROB .LE. 0) GO TO 30
         FACTOR = 1.0D0
         DO 20 K = 1, NTRIES
            CALL INITPT(N,X,NPROB,FACTOR)
            WRITE (91,60) NPROB,N,M,FACTOR
            NFEV = 0
            NJEV = 0
            CALL LMDER1(FCN,M,N,X,FVEC,FJAC,LDFJAC,TOL,INFO,IWA,WA,
     *                  LWA)
            CALL SSQFCN(M,N,X,FVEC,NPROB)
            FNORM2 = ENORM(M,FVEC)
            WRITE (91,70) INFO,NFEV,NJEV,FNORM2,(X(I), I = 1, N)
            FACTOR = 1.0D1*FACTOR
   20       CONTINUE
         GO TO 10
   30 CONTINUE
      STOP
   50 FORMAT (4I5)
   60 FORMAT ('C ',3I5,1X,ES25.17E3)
   70 FORMAT ('F ',3I6,99(1X,ES25.17E3))
      END
