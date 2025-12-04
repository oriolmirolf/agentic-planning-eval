(define (domain loadbalance)
  (:requirements :strips :typing :negative-preconditions :equality :fluents)

  (:types
    server region service bucket
  )

  (:predicates
    (server-region ?s - server ?r - region)
    (bucket-region ?b - bucket ?r - region)
    (bucket-service ?b - bucket ?svc - service)
    (online ?s - server)
    (runs ?s - server ?svc - service)
    (assigned-to ?b - bucket ?s - server)
    (unassigned ?b - bucket)
  )

  (:functions
    (capacity ?s - server)
    (load ?s - server)
  )

  (:action power-on
    :parameters (?s - server)
    :precondition (not (online ?s))
    :effect (online ?s)
  )

  (:action power-off
    :parameters (?s - server)
    :precondition (and
      (online ?s)
      (= (load ?s) 0)
    )
    :effect (not (online ?s))
  )

  (:action start-service
    :parameters (?s - server ?svc - service)
    :precondition (and
      (online ?s)
      (not (runs ?s ?svc))
    )
    :effect (runs ?s ?svc)
  )

  (:action stop-service
    :parameters (?s - server ?svc - service)
    :precondition (and
      (online ?s)
      (runs ?s ?svc)
    )
    :effect (not (runs ?s ?svc))
  )

  (:action assign-bucket
    :parameters (?b - bucket ?s - server ?r - region ?svc - service)
    :precondition (and
      (unassigned ?b)
      (online ?s)
      (bucket-region ?b ?r)
      (server-region ?s ?r)
      (bucket-service ?b ?svc)
      (runs ?s ?svc)
      (< (load ?s) (capacity ?s))
    )
    :effect (and
      (not (unassigned ?b))
      (assigned-to ?b ?s)
      (increase (load ?s) 1)
    )
  )

  (:action unassign-bucket
    :parameters (?b - bucket ?s - server)
    :precondition (assigned-to ?b ?s)
    :effect (and
      (unassigned ?b)
      (not (assigned-to ?b ?s))
      (decrease (load ?s) 1)
    )
  )
)
