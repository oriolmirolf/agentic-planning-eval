(define (domain rush-hour)
    (:requirements :typing :strips)
    (:types position car - object)

    (:predicates
        (ADJACENT ?p1 - position ?p2 - position ?p3 - position)
        (SMALL ?c - car)
        (LARGE ?c - car)
        (isOccupied ?p - position)
        (containsCar ?p - position ?c - car)
    )

    ; the precodintions for moving a small car:
    ; 1- car should be small
    ; 2- the next position p3 should be ADJACENT to current postions p1 and p2
    ; 3- car should be placed in postions: p1 and p2
    ; 4- the next postion p3 where the car will be placed should be empty
    
    ;the effects of moving the small car:
    ;1- the previous position p1 is no longer occupied
    ;2- position p1 doesn't contain the car anymore
    ;3- the next position p3 will contain the car
    ;4- position p3 is now occupied
    (:action move-small
        :parameters (?c - car ?p1 - position ?p2 - position ?p3 - position)
        :precondition (and 
            (SMALL ?c) 
            (ADJACENT ?p1 ?p2 ?p3)
            (containsCar ?p1 ?c)
            (containsCar ?p2 ?c)
            (not (isOccupied ?p3))
        )
        :effect (and
            (not (isOccupied ?p1))
            (not (containsCar ?p1 ?c))
            (containsCar ?p3 ?c)
            (isOccupied ?p3)
        )
    )

    ; the precodintions for moving a large car:
    ; 1- car should be large
    ; 2- the next position p4 should be ADJACENT to current postions p1,p2 and p3
    ; 3- car should be placed in postions: p1,p2,p3
    ; 4- the next postion p4 where the car will be placed should be empty
    
    ;the effects of moving the small car:
    ;1- the previous position p1 is no longer occupied
    ;2- position p1 doesn't contain the car anymore
    ;3- the next position p4 will contain the car
    ;4- position p4 is now occupied
    
    (:action move-large
        :parameters (?c - car ?p1 - position ?p2 - position ?p3 - position ?p4 - position)
        :precondition (and
            (LARGE ?c) 
            (ADJACENT ?p1 ?p2 ?p3) (ADJACENT ?p2 ?p3 ?p4)
            (containsCar ?p1 ?c)
            (containsCar ?p2 ?c)
            (containsCar ?p3 ?c)
            (not (isOccupied ?p4))
        )
        :effect (and
            (not (isOccupied ?p1))
            (not (containsCar ?p1 ?c))
            (containsCar ?p4 ?c)
            (isOccupied ?p4)
        )
    )
)