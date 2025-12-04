(define (domain manufacturing)
  (:requirements :strips :typing :action-costs)
  (:types part machine)

  (:predicates
    ;; State of parts
    (raw ?p - part)
    (cut ?p - part)
    (drilled ?p - part)
    (painted ?p - part)

    ;; Machine capabilities and availability
    (available ?m - machine)
    (can-cut ?m - machine)
    (can-drill ?m - machine)
    (can-paint ?m - machine)
  )

  ;; Total action cost
  (:functions (total-cost))

  (:action cut-part
    :parameters (?p - part ?m - machine)
    :precondition (and
      (raw ?p)
      (available ?m)
      (can-cut ?m)
    )
    :effect (and
      (cut ?p)
      (not (raw ?p))
      (increase (total-cost) 1)
    )
  )

  (:action drill-part
    :parameters (?p - part ?m - machine)
    :precondition (and
      (cut ?p)
      (available ?m)
      (can-drill ?m)
    )
    :effect (and
      (drilled ?p)
      (increase (total-cost) 1)
    )
  )

  (:action paint-part
    :parameters (?p - part ?m - machine)
    :precondition (and
      (drilled ?p)
      (available ?m)
      (can-paint ?m)
    )
    :effect (and
      (painted ?p)
      (increase (total-cost) 1)
    )
  )
)
