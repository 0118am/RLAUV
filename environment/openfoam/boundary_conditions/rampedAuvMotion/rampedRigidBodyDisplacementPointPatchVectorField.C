#include "rampedRigidBodyDisplacementPointPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "pointPatchFields.H"
#include "polyMesh.H"
#include "Time.H"

namespace Foam
{

rampedRigidBodyDisplacementPointPatchVectorField::
rampedRigidBodyDisplacementPointPatchVectorField
(
    const pointPatch& patch,
    const DimensionedField<vector, pointMesh>& internalField
)
:
    fixedValuePointPatchField<vector>(patch, internalField),
    motionKind_("translation"),
    axis_(vector(1, 0, 0)),
    origin_(Zero),
    amplitude_(0),
    omega_(0),
    phase_(0),
    rampDuration_(0),
    p0_(patch.localPoints())
{}


rampedRigidBodyDisplacementPointPatchVectorField::
rampedRigidBodyDisplacementPointPatchVectorField
(
    const pointPatch& patch,
    const DimensionedField<vector, pointMesh>& internalField,
    const dictionary& dict
)
:
    fixedValuePointPatchField<vector>(patch, internalField, dict),
    motionKind_(dict.get<word>("motionKind")),
    axis_(dict.get<vector>("axis")),
    origin_(dict.get<point>("origin")),
    amplitude_(dict.get<scalar>("amplitude")),
    omega_(dict.get<scalar>("omega")),
    phase_(dict.get<scalar>("phase")),
    rampDuration_(dict.get<scalar>("rampDuration")),
    p0_(patch.localPoints())
{
    // The fixed-value base has already read the displacement.  Recover the
    // undeformed points from the current mesh on restart instead of writing a
    // second full point field named p0 into every time directory.
    p0_ = patch.localPoints() - vectorField(*this);
    if (!dict.found("value"))
    {
        updateCoeffs();
    }
}


rampedRigidBodyDisplacementPointPatchVectorField::
rampedRigidBodyDisplacementPointPatchVectorField
(
    const rampedRigidBodyDisplacementPointPatchVectorField& source,
    const pointPatch& patch,
    const DimensionedField<vector, pointMesh>& internalField,
    const pointPatchFieldMapper& mapper
)
:
    fixedValuePointPatchField<vector>(source, patch, internalField, mapper),
    motionKind_(source.motionKind_),
    axis_(source.axis_),
    origin_(source.origin_),
    amplitude_(source.amplitude_),
    omega_(source.omega_),
    phase_(source.phase_),
    rampDuration_(source.rampDuration_),
    p0_(source.p0_, mapper)
{}


rampedRigidBodyDisplacementPointPatchVectorField::
rampedRigidBodyDisplacementPointPatchVectorField
(
    const rampedRigidBodyDisplacementPointPatchVectorField& source,
    const DimensionedField<vector, pointMesh>& internalField
)
:
    fixedValuePointPatchField<vector>(source, internalField),
    motionKind_(source.motionKind_),
    axis_(source.axis_),
    origin_(source.origin_),
    amplitude_(source.amplitude_),
    omega_(source.omega_),
    phase_(source.phase_),
    rampDuration_(source.rampDuration_),
    p0_(source.p0_)
{}


void rampedRigidBodyDisplacementPointPatchVectorField::autoMap
(
    const pointPatchFieldMapper& mapper
)
{
    fixedValuePointPatchField<vector>::autoMap(mapper);
    p0_.autoMap(mapper);
}


void rampedRigidBodyDisplacementPointPatchVectorField::rmap
(
    const pointPatchField<vector>& field,
    const labelList& addresses
)
{
    const rampedRigidBodyDisplacementPointPatchVectorField& source =
        refCast<const rampedRigidBodyDisplacementPointPatchVectorField>(field);
    fixedValuePointPatchField<vector>::rmap(field, addresses);
    p0_.rmap(source.p0_, addresses);
}


void rampedRigidBodyDisplacementPointPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    const scalar timeValue = this->internalField().mesh()().time().value();
    scalar ramp = 1.0;
    if (rampDuration_ > SMALL && timeValue < rampDuration_)
    {
        const scalar x = max(scalar(0), min(scalar(1), timeValue/rampDuration_));
        ramp = x*x*x*(10.0 + x*(-15.0 + 6.0*x));
    }
    const scalar coordinate = amplitude_*ramp*sin(omega_*timeValue + phase_);
    const vector axisHat = axis_/mag(axis_);

    if (motionKind_ == "translation")
    {
        vectorField::operator=(axisHat*coordinate);
    }
    else if (motionKind_ == "rotation")
    {
        const vectorField p0Relative(p0_ - origin_);
        vectorField::operator=
        (
            p0Relative*(cos(coordinate) - 1.0)
          + (axisHat ^ p0Relative*sin(coordinate))
          + (axisHat & p0Relative)*(1.0 - cos(coordinate))*axisHat
        );
    }
    else
    {
        FatalErrorInFunction
            << "motionKind must be translation or rotation, got " << motionKind_
            << exit(FatalError);
    }

    fixedValuePointPatchField<vector>::updateCoeffs();
}


void rampedRigidBodyDisplacementPointPatchVectorField::write(Ostream& stream) const
{
    pointPatchField<vector>::write(stream);
    stream.writeEntry("motionKind", motionKind_);
    stream.writeEntry("axis", axis_);
    stream.writeEntry("origin", origin_);
    stream.writeEntry("amplitude", amplitude_);
    stream.writeEntry("omega", omega_);
    stream.writeEntry("phase", phase_);
    stream.writeEntry("rampDuration", rampDuration_);
    this->writeValueEntry(stream);
}


makePointPatchTypeField
(
    pointPatchVectorField,
    rampedRigidBodyDisplacementPointPatchVectorField
);

}
