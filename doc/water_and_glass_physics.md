# Roblet Underwater Simulation — Physics Parameter Reasoning

The reasoning for the fluid, contact, and geom parameters used in the MuJoCo
stick-slip paddling simulation.

---

## 1. Global Simulation Environment (`<option>`)

```xml
<option density="1000" viscosity="0.0009" gravity="0 0 -9.81" integrator="implicitfast"/>
```

| Parameter | Purpose |
|---|---|
| `density="1000"` | Matches water. MuJoCo uses this with each geom's fluid-equivalent volume to compute a continuous Archimedes buoyancy force opposing gravity. |
| `viscosity="0.0009"` | Dynamic viscosity of water at ~25°C; sets the baseline (non-shape-dependent) part of drag. |
| `gravity="0 0 -9.81"` | Standard gravity. Net submerged weight = gravity − buoyancy. |
| `integrator="implicitfast"` | Fluid drag is velocity-dependent. Explicit Euler evaluates forces from the previous step only, so at `0.01s` timestep, stacking drag with the magnetic torque callback can produce numerical overshoot/jitter. `implicitfast` solves the velocity-dependent terms implicitly, giving stability at the same timestep without the 10x cost of shrinking it. |

---

## 2. Glass Floor (`<asset>` & `<worldbody>`)

```xml
<material name="submerged_glass" rgba="0.6 0.8 0.9 0.4" shininess="0.9" specular="1"/>
<geom name="wet_glass_floor" type="plane" size="0 0 0.1" material="submerged_glass"
      friction="0.25 0.005 0.0001" solimp="0.9 0.95 0.001 0.5 2" solref="0.01 1" condim="3"/>
```

- **`friction="0.25 0.005 0.0001"`** — lowered sliding friction models lubrication from the
  trapped water film between the resin body and glass; the low torsional/rolling terms let the
  robot pivot without sticking.
- **`solimp="0.9 0.95 0.001 0.5 2"`** — 5 required values (`dmin, dmax, width, midpoint, power`).
  `dmin=0.9` softens the constraint at small penetration to avoid velocity spikes on contact;
  `dmax=0.95` stiffens it near max penetration to resist tunneling through rigid glass.
- **`solref="0.01 1"`** — 10 ms time constant, critical damping (1.0); resolves the fast 0.1 s
  snapback impact cleanly without artificial bounce.
- **`condim="3"`** — normal force + 2 tangential friction axes; the minimum needed for Coulomb
  stick/slip.
- Static plane bodies have infinite effective mass, so no `fluidshape` is needed here —
  "wet" is fully expressed through the lowered friction, correctly.

---

## 3. Robotic Module (`<geom>`)

```xml
<geom name="geom_Body_1" type="mesh" mesh="Body" rgba="0.2 0.2 0.8 1"
      fluidshape="ellipsoid" density="1200" fluidcoef="0.5 0.25 1.5 1.0 1.0"/>
```

- **`density="1200"`** — matches SLA resin (~1180–1200 kg/m³), denser than water by
  ~200 kg/m³. That net difference sets the submerged weight, which sets the normal force at
  the floor contact, which sets the Coulomb friction ceiling driving stick-slip.
- **`fluidshape="ellipsoid"`** — without this flag, MuJoCo estimates fluid forces from the
  *body's* equivalent inertia box (mass-distribution-derived, applied once per body), which is
  coarser and less accurate for irregular meshes than a per-geom fit. Turning it on switches to
  the Boyer et al. model, which maps each geom to a best-fit ellipsoid and computes separate
  blunt-drag, slender-drag, angular-drag, and lift terms depending on the geom's orientation
  relative to its velocity — a flat face pushing broadside sees far more resistance than the
  same face slicing edge-on.
- **`fluidcoef="0.5 0.25 1.5 1.0 1.0"`** — leave at MuJoCo's defaults rather than hand-tuning
  `blunt_drag` upward. This coefficient is a fixed shape property of the geom; it isn't
  something to adjust based on magnet count or placement. The actual stroke asymmetry
  (slow 0.9 s tilt-down "stick" vs. fast 0.1 s tilt-up "slip") comes for free from the drag
  model, since drag scales with velocity: the faster angular velocity in the snapback phase
  produces higher instantaneous drag opposing that motion, and the geometry naturally presents
  a different cross-section to the flow at each point in the stroke. If you do want to tune
  `fluidcoef`, do it by matching measured or paper-reported torque/speed curves (e.g. the
  13.2 mm/step figure at 0.5 Hz step-out), not by reasoning backward from the magnet layout.

---