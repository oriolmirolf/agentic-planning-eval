(define (problem logistics-p05)
  (:domain logistics)
  (:objects
    city1 city2 - city
    loc1 airport1 loc2 airport2 - location
    truck1 truck2 - truck
    plane1 - airplane
    p1 p2 p3 p4 p5 - package
  )
  (:init
    (= (total-cost) 0)

    ;; City membership
    (in-city loc1 city1)
    (in-city airport1 city1)
    (in-city loc2 city2)
    (in-city airport2 city2)

    ;; Airports
    (airport airport1)
    (airport airport2)

    ;; Vehicle locations
    (at-truck truck1 loc1)
    (at-truck truck2 airport2)
    (at-airplane plane1 airport1)

    ;; Packages
    (at-package p1 loc1)
    (at-package p2 loc1)
    (at-package p3 loc1)
    (at-package p4 loc1)
    (at-package p5 loc1)
  )
  (:goal (and
    (at-package p1 loc2)
    (at-package p2 loc2)
    (at-package p3 loc2)
    (at-package p4 loc2)
    (at-package p5 loc2)
  ))
  (:metric minimize (total-cost))
)
