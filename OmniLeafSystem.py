from pydrake.systems.framework import LeafSystem_
from pydrake.systems.scalar_conversion import TemplateSystem


import numpy as np
import matplotlib.pyplot as plt
from params import Params


@TemplateSystem.define("OmniLeafSystem_")
def OmniLeafSystem_(T):
    class Impl(LeafSystem_[T]):
        def _construct(self, converter=None, params=Params()):
            LeafSystem_[T].__init__(self, converter=converter)
            self._params = params

            # Continuous state
            self.DeclareContinuousState(3)  # [x, y, theta]

            # Input
            self.DeclareVectorInputPort("wheel", 3)  # [v1, v2, v3]

            # Output
            self.DeclareVectorOutputPort(
                "state", 3, self.CopyStateOut)  # [x, y, theta]

        def _construct_copy(self, other, converter=None):
            Impl._construct(self, converter=converter, params=other._params)

        def DoCalcTimeDerivatives(self, context, derivatives):
            v1 = self.EvalVectorInput(context, 0).GetAtIndex(0)
            v2 = self.EvalVectorInput(context, 0).GetAtIndex(1)
            v3 = self.EvalVectorInput(context, 0).GetAtIndex(2)
            theta = context.get_continuous_state_vector().GetAtIndex(2)

            v_heading = np.sqrt(3) * (v3 - v1) / 3
            v_normal = (v3 + v1) / 3 - (2/3) * v2
            omega = (1 / (3 * self._params.d)) * (v1 + v2 + v3)

            x_dot = v_heading * np.cos(theta) + v_normal * np.sin(theta)
            y_dot = v_heading * np.sin(theta) - v_normal * np.cos(theta)
            theta_dot = omega

            derivatives.get_mutable_vector().SetAtIndex(0, x_dot)
            derivatives.get_mutable_vector().SetAtIndex(1, y_dot)
            derivatives.get_mutable_vector().SetAtIndex(2, theta_dot)

        def CopyStateOut(self, context, output):
            state = context.get_continuous_state_vector().CopyToVector()
            output.SetFromVector(state)

    return Impl
