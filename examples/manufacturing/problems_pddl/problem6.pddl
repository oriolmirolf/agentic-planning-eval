(define (problem manufacturing-problem-6)
  (:domain manufacturing)

  (:objects
    p1 p2 p3 p4 p5 - part
    cutterA combo1 paintbooth1 - machine
  )

  (:init
    (raw p1)
    (raw p2)
    (raw p3)
    (raw p4)
    (raw p5)

    (available cutterA)
    (available combo1)
    (available paintbooth1)

    (can-cut cutterA)
    (can-cut combo1)
    (can-drill combo1)
    (can-paint paintbooth1)

    (= (total-cost) 0)
  )

  (:goal
    (and
      (painted p1)  ;; full pipeline
      (painted p2)  ;; full pipeline
      (drilled p3)  ;; only up to drilled
      (cut p4)      ;; only cut
      ;; p5 has no requirement
    )
  )

  (:metric minimize (total-cost))
)
