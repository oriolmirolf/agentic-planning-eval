(define (problem manufacturing-problem-3)
  (:domain manufacturing)

  (:objects
    p1 p2 p3 - part
    cutter1 drill1 paintbooth1 - machine
  )

  (:init
    (raw p1)
    (raw p2)
    (raw p3)

    (available cutter1)
    (available drill1)
    (available paintbooth1)

    (can-cut cutter1)
    (can-drill drill1)
    (can-paint paintbooth1)

    (= (total-cost) 0)
  )

  (:goal
    (and
      (painted p1)  ;; full pipeline
      (drilled p2)  ;; only up to drilled
      (cut p3)      ;; only cut
    )
  )

  (:metric minimize (total-cost))
)
