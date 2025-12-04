(define (problem manufacturing-problem-5)
  (:domain manufacturing)

  (:objects
    p1 p2 p3 p4 - part
    cutter1 drill1 paintbooth1 - machine
  )

  (:init
    ;; processing state
    (raw p1)
    (raw p2)
    (cut p3)
    (cut p4)
    (drilled p4)

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
      (drilled p2)  ;; cut + drill needed
      (painted p3)  ;; cut already, needs drill + paint
      (painted p4)  ;; cut+drilled already, needs paint
    )
  )

  (:metric minimize (total-cost))
)
