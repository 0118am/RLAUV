#ifndef rampedRigidBodyDisplacementPointPatchVectorField_H
#define rampedRigidBodyDisplacementPointPatchVectorField_H

#include "fixedValuePointPatchField.H"

namespace Foam
{

class rampedRigidBodyDisplacementPointPatchVectorField
:
    public fixedValuePointPatchField<vector>
{
    word motionKind_;
    vector axis_;
    point origin_;
    scalar amplitude_;
    scalar omega_;
    scalar phase_;
    scalar rampDuration_;
    pointField p0_;

public:
    TypeName("rampedRigidBodyDisplacement");

    rampedRigidBodyDisplacementPointPatchVectorField
    (
        const pointPatch&,
        const DimensionedField<vector, pointMesh>&
    );

    rampedRigidBodyDisplacementPointPatchVectorField
    (
        const pointPatch&,
        const DimensionedField<vector, pointMesh>&,
        const dictionary&
    );

    rampedRigidBodyDisplacementPointPatchVectorField
    (
        const rampedRigidBodyDisplacementPointPatchVectorField&,
        const pointPatch&,
        const DimensionedField<vector, pointMesh>&,
        const pointPatchFieldMapper&
    );

    rampedRigidBodyDisplacementPointPatchVectorField
    (
        const rampedRigidBodyDisplacementPointPatchVectorField&,
        const DimensionedField<vector, pointMesh>&
    );

    virtual autoPtr<pointPatchField<vector>> clone() const
    {
        return pointPatchField<vector>::Clone(*this);
    }

    virtual autoPtr<pointPatchField<vector>> clone
    (
        const DimensionedField<vector, pointMesh>& internalField
    ) const
    {
        return pointPatchField<vector>::Clone(*this, internalField);
    }

    virtual void autoMap(const pointPatchFieldMapper&);
    virtual void rmap(const pointPatchField<vector>&, const labelList&);
    virtual void updateCoeffs();
    virtual void write(Ostream&) const;
};

}

#endif
