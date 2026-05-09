# The Scenario: Dealing With Lab Safety

## Overview

A facility at the Georgia Tech Research Institute (GTRI) has recently experienced a minor accident. Fortunately, no one was injured, but the incident has raised concerns about overall lab safety. In response, the lab’s management wants to understand the root causes of such accidents so they can take preventative measures. You, being a savvy computer science student, have been hired as a safety consultant to model the situation with the given information as a Bayesian network.

For this part of the assignment, your task is to design a Bayesian network that represents the following events and their interdependencies. Your network will help estimate the likelihood of severe damage (an outcome that could have endangered the researchers and the expensive equipment) under various circumstances.

## The Events

### Training
Lab personnel must receive proper training. Unfortunately, the lab has started to reallocate its budget towards their snacks and coffee fund (they _really_ like their snacks). As such, there is a 0.8 probability that staff are properly trained (call this true). 

### Inspection
The lab is subject to regular safety inspections. Unfortunately, the researchers have spent too much time hanging out by the water cooler, so they sometimes fail the inspection. There is a 0.7 probability that the lab does pass inspection (call this true). 

### Failure of Equipment
Because the lab has spent so much money on snacks and coffee, some of their equipment has started to show their age, and has started to fail intermittently, with a 0.8 probability of **not** failing (call this false). 

### Response Readiness
The emergency response team's readiness can help mitigate an incident. However, with the addition of the new snack bar, the team has started to slack off, and are only on guard (call this true) half the time, with a probability of 0.5. 

### Adherence to Protocol
The degree to which lab personnel follow the safety protocols depends on both their training, and whether the lab passed inspection. 
* If the lab passed inspection and the lab members have received proper training, the staff does a *really* good job of following protocols, with a 0.95 probability of adherence (call this true).
* If the lab personnel has received their training, but the lab has not passed inspection, the staff works diligently, but can't cover up all the mistakes, so the probability of **not** adhering to protocol is 0.25 (call this false). 
* If the lab has passed inspections, but the administrators decide to slack off and not train the lab personnel, the staff is not nearly as diligent, and there is a 0.5 probability of adhering to the safety protocols (call this true).
* If the lab has not passed inspections AND the lab personnel have not received their training, the lab members are just sitting ducks and there is only a 0.1 probability (call this true) of adhering to the safety protocols. They're in for a bad time.

### Accident Occurrence
An accident may occur due to a combination of equipment failure and whether the response team is ready. 
* If the equipment is working and the response team is ready, the probability of an accident (call this true) is quite low, at 0.1.
* If equipment is working, but the response team is *not* ready (because they're chowing down on all the new munchies), the chance of an accident rises (call this true) to 0.2. 
* If the equipment fails, but the response team is ready, because the equipment is quite old, the probability of an accident (call this true) still rises to 0.40. 
* If the equipment fails AND the response team is not ready, the probability of an accident (call this true) sharply increases to 0.80. OSHA would not be pleased.

### Severity of Damage
Finally, the severity of the damage caused by an accident depends on whether an accident even happens, and the level of adherence to safety protocols.
* If protocols are not followed and no accident occurs, the chance of severe damage is unlikely, but still possible (call this true) at a probability of 0.10. 
* If protocols are not followed AND an accident occurs, the probability of severe damage (call this true) is quite high, at 0.95. 
* If protocols are followed and no accident occurs, everyone is happy, because there is a 0.95 probability that there is no severe damage (call this false). But the lab must stay vigilant! 
* If protocols are followed, but an accident still happens (say, the beloved snack bar spontaneously combusts), the probability of severe damage (call this true) is 0.40. 