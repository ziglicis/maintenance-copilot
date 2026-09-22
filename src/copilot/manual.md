# Turbofan sustainment manual, extract

**This document is synthetic.** It was written for this demonstration project. It is
not a real maintenance manual, it describes no real engine, and nothing in it should
be used to maintain actual hardware. Part numbers and task codes are invented.

## Scope

Condition-based inspection guidance for the simulated high-bypass turbofan used in
the C-MAPSS dataset. Covers four modules: fan, high pressure compressor (HPC), high
pressure turbine (HPT) and low pressure turbine (LPT).

## Sensor to module map

| Sensor | Description | Module |
|---|---|---|
| s2 (T24) | Total temperature at LPC outlet | Fan / LPC |
| s3 (T30) | Total temperature at HPC outlet | HPC |
| s4 (T50) | Total temperature at LPT outlet | LPT |
| s7 (P30) | Total pressure at HPC outlet | HPC |
| s8 (Nf) | Physical fan speed | Fan |
| s9 (Nc) | Physical core speed | HPC / HPT |
| s11 (Ps30) | Static pressure at HPC outlet | HPC |
| s12 (phi) | Fuel flow to Ps30 ratio | Combustor / HPC |
| s13 (NRf) | Corrected fan speed | Fan |
| s14 (NRc) | Corrected core speed | HPC / HPT |
| s15 (BPR) | Bypass ratio | Fan |
| s17 (htBleed) | Bleed enthalpy | HPC |
| s20 (W31) | HPT coolant bleed flow | HPT |
| s21 (W32) | LPT coolant bleed flow | LPT |

## Degradation signatures

**HPC efficiency loss.** The dominant failure mode in this fleet. Presents as rising
HPC outlet temperature (s3) and rising static pressure (s11), with corrected core
speed (s14) drifting up as the control system compensates. Bleed enthalpy (s17)
usually rises with it. Fuel burn increases before any thrust shortfall is felt.

**HPT blade and seal deterioration.** Coolant bleed flow (s20) falls as seals wear
and clearances open. LPT outlet temperature (s4) rises because more energy passes
downstream. Often follows or accompanies HPC degradation.

**LPT efficiency loss.** LPT outlet temperature (s4) rises while LPT coolant bleed
(s21) falls. Bypass ratio (s15) drifts.

**Fan module wear.** Physical and corrected fan speed (s8, s13) diverge slowly from
their commanded values, and LPC outlet temperature (s2) rises. Slowest of the four
signatures to develop.

## Inspection tasks

| Task | Module | Steps | Typical duration |
|---|---|---|---|
| TSK-201 borescope, HPC stages 1 to 9 | HPC | Access through port B2. Inspect blade tips for rub, leading edges for erosion, and vanes for distortion. Photograph any indication over 1 mm. | 4 h |
| TSK-214 HPC efficiency trend review | HPC | Pull last 30 cycles of s3, s11, s17. Compare against the module baseline recorded at last overhaul. | 1 h |
| TSK-330 borescope, HPT stage 1 | HPT | Access through port T1. Check for thermal barrier coating loss, tip burn and cooling hole blockage. | 5 h |
| TSK-338 HPT clearance and seal check | HPT | Measure blade tip clearance at four stations. Inspect honeycomb seal for wear. | 6 h |
| TSK-410 LPT borescope and gas path inspection | LPT | Inspect stages 3 to 5 for cracking and distortion. | 4 h |
| TSK-505 fan blade and containment inspection | Fan | Visual and dye penetrant on all fan blades. Check root fixings. | 3 h |
| TSK-520 gas path performance run | All | Ground run at three power settings. Record all gas path sensors. Compare with acceptance limits. | 2 h |

## Common parts

| Part | Number | Module | Lead time |
|---|---|---|---|
| HPC stage 5 blade set | P-HPC-5051 | HPC | 21 days |
| HPC variable vane bushing kit | P-HPC-2210 | HPC | 10 days |
| HPT stage 1 blade set | P-HPT-1100 | HPT | 35 days |
| HPT honeycomb seal segment | P-HPT-1145 | HPT | 28 days |
| HPT nozzle guide vane set | P-HPT-1180 | HPT | 30 days |
| LPT stage 4 blade set | P-LPT-4040 | LPT | 25 days |
| Fan blade, single | P-FAN-0012 | Fan | 14 days |
| Gas path seal kit, general | P-GEN-0900 | All | 7 days |

## Priority guidance

| Priority | Condition | Expected response |
|---|---|---|
| Immediate | Predicted remaining life under 15 cycles, or lower interval bound under 10 | Ground the aircraft at the next landing. Do not release for a further mission. |
| Urgent | Predicted remaining life 15 to 30 cycles | Schedule within the next 10 cycles. Order long lead parts now. |
| Routine | Predicted remaining life 30 to 60 cycles | Fold into the next scheduled maintenance window. |
| Monitor | Predicted remaining life over 60 cycles | No action. Continue trend monitoring. |

## Notes for planners

Parts lead time, not engineering time, is what usually drives the schedule. An HPT
blade set at 35 days is longer than the remaining life of an engine flagged at 30
cycles on a two-sortie-per-day tasking, so long lead items should be staged on the
first Urgent flag rather than on the pull decision.
