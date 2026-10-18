; Skein demo: closures, recursion, lists, and proper tail calls -- all compiled
; to C through knotkit.

(define (fact n)
  (if (= n 0) 1 (* n (fact (- n 1)))))

; Tail-recursive: a million iterations run in constant stack (the trampoline).
(define (sum-to n acc)
  (if (= n 0) acc (sum-to (- n 1) (+ acc n))))

(define (range a b)
  (if (= a b) (quote ()) (cons a (range (+ a 1) b))))

(define (map f xs)
  (if (null? xs) (quote ()) (cons (f (car xs)) (map f (cdr xs)))))

(define (adder n) (lambda (x) (+ x n)))

(display (fact 10))                 (newline)   ; 3628800
(display (sum-to 1000000 0))        (newline)   ; 500000500000  (tail recursion)
(display (map (adder 100) (range 0 5)))         ; (100 101 102 103 104)
(newline)
