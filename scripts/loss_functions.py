from math import sqrt,log,atan2,exp,cosh
from pickle import FALSE
from re import S
import sys
sys.path.append('../build')
import GlobalSfMpy as sfm

# For a residual vector with squared 2-norm 'sq_norm', this method
# is required to fill in the value and derivatives of the loss
# function (rho in this example):
#
#   out[0] = rho(sq_norm),
#   out[1] = rho'(sq_norm),
#   out[2] = rho''(sq_norm),
#
# Here the convention is that the contribution of a term to the
# cost function is given by 1/2 rho(s),  where
#
#   s = ||residuals||^2.
#
# Calling the method with a negative value of 's' is an error and
# the implementations are not required to handle that case.
#
# Most sane choices of rho() satisfy:
#
#   rho(0) = 0,
#   rho'(0) = 1,
#   rho'(s) < 1 in outlier region,
#   rho''(s) < 0 in outlier region,
#
# so that they mimic the least squares cost for small residuals.

# Scaling
# -------
# Given one robustifier
#   s -> rho(s)
# one can change the length scale at which robustification takes
# place, by adding a scale factor 'a' as follows:
#
#   s -> a^2 rho(s / a^2).
#
# The first and second derivatives are:
#
#   s -> rho'(s / a^2),
#   s -> (1 / a^2) rho''(s / a^2),

class TrivialLoss(sfm.LossFunction):
    # rho(s)=s
    def __init__(self):
        super().__init__()
    def Evaluate(self, sq_norm, out):
        out[0] = sq_norm;
        out[1] = 1.0
        out[2] = 0.0

class HuberLoss(sfm.LossFunction):
    #   rho(s) = s               for s <= 1,
    #   rho(s) = 2 sqrt(s) - 1   for s >= 1.
    def __init__(self,a):
        self.a = a
        self.b = a*a
        super().__init__()
    def Evaluate(self, sq_norm, out):
        if sq_norm > self.b:
            r = sqrt(sq_norm)
            out[0] = 2*self.a*r - self.b
            out[1] = max(self.a/r, sys.float_info.min)
            out[2] = -out[1]/(2.0*sq_norm)
        else:
            out[0] = sq_norm
            out[1] = 1.0
            out[2] = 0.0
            
class SoftLOneLoss(sfm.LossFunction):
    #   rho(s) = 2 (sqrt(1 + s) - 1).
    def __init__(self,a):
        self.a = a
        self.b = a*a
        self.c = 1.0/self.b
        super().__init__()
    def Evaluate(self, sq_norm, out):
        sum = 1.0 + sq_norm*self.c
        tmp = sqrt(sum)
        out[0] = 2.0 * self.b * (tmp - 1.0)
        out[1] = max(1.0/tmp, sys.float_info.min)
        out[2] = -(self.c*out[1])/(2.0*sum)

class CauchyLoss(sfm.LossFunction):
    #   rho(s) = log(1 + s).
    def __init__(self,a):
        self.b = a*a
        self.c = 1.0/self.b
        super().__init__()
    def Evaluate(self, sq_norm, out):
        sum = 1.0 + sq_norm*self.c
        inv = 1.0/sum
        out[0] = self.b * log(sum)
        out[1] = max(inv, sys.float_info.min)
        out[2] = -self.c * (inv * inv)
        
class ArctanLoss(sfm.LossFunction):
    #   rho(s) = a arctan(s / a).
    def __init__(self,a):
        self.a = a
        self.b = 1 / (a*a)
        super().__init__()
    def Evaluate(self, sq_norm, out):
        sum = 1 + sq_norm*sq_norm*self.b
        inv = 1.0/sum
        out[0] = self.a * atan2(sq_norm, self.a)
        out[1] = max(inv, sys.float_info.min)
        out[2] = -2.0 * sq_norm * self.b * (inv * inv)

class TolerantLoss(sfm.LossFunction):
    # Loss function that maps to approximately zero cost in a range around the
    # origin, and reverts to linear in error (quadratic in cost) beyond this range.
    # The tolerance parameter 'a' sets the nominal point at which the
    # transition occurs, and the transition size parameter 'b' sets the nominal
    # distance over which most of the transition occurs. Both a and b must be
    # greater than zero, and typically b will be set to a fraction of a.
    # The slope rho'[s] varies smoothly from about 0 at s <= a - b to
    # about 1 at s >= a + b.
    #
    # The term is computed as:
    #
    #   rho(s) = b log(1 + exp((s - a) / b)) - c0.
    #
    # where c0 is chosen so that rho(0) == 0
    #
    #   c0 = b log(1 + exp(-a / b)
    #
    # This has the following useful properties:
    #
    #   rho(s) == 0               for s = 0
    #   rho'(s) ~= 0              for s << a - b
    #   rho'(s) ~= 1              for s >> a + b
    #   rho''(s) > 0              for all s
    #
    # In addition, all derivatives are continuous, and the curvature is
    # concentrated in the range a - b to a + b.
    #
    # At s = 0: rho = [0, ~0, ~0].
    def __init__(self,a,b):
        self.a = a
        self.b = b
        self.c = b * log(1 + exp(-a / b))
        assert(a >= 0)
        assert(b > 0)
        super().__init__()
    def Evaluate(self, sq_norm, out):
        x = (sq_norm - self.a) / self.b;
        # The basic equation is rho[0] = b ln(1 + e^x).  However, if e^x is too
        # large, it will overflow.  Since numerically 1 + e^x == e^x when the
        # x is greater than about ln(2^53) for doubles, beyond this threshold
        # we substitute x for ln(1 + e^x) as a numerically equivalent approximation.
        kLog2Pow53 = 36.7;  # ln(MathLimits<double>::kEpsilon).
        if (x > kLog2Pow53):
            out[0] = sq_norm - self.a - self.c;
            out[1] = 1.0;
            out[2] = 0.0;
        else:
            e_x = exp(x);
            out[0] = self.b * log(1.0 + e_x) - self.c;
            out[1] =  max(e_x / (1.0 + e_x), sys.float_info.min);
            out[2] = 0.5 / (self.b * (1.0 + cosh(x)));
        
class TukeyLoss(sfm.LossFunction):
    #   rho(s) = a^2 / 6 * (1 - (1 - s / a^2)^3 )   for s <= a^2,
    #   rho(s) = a^2 / 6                            for s >  a^2.
    def __init__(self,a):
        self.a_squared = a*a
        super().__init__()
    def Evaluate(self, sq_norm, out):
        if sq_norm <= self.a_squared:
            # Inlier region.
            value = 1.0 - sq_norm / self.a_squared;
            value_sq = value * value;
            out[0] = self.a_squared / 6.0 * (1.0 - value_sq * value);
            out[1] = 0.5 * value_sq;
            out[2] = -1.0 / self.a_squared * value;
        else:
            # Outlier region.
            out[0] = self.a_squared / 6.0;
            out[1] = 0.0;
            out[2] = 0.0;
            
class LOneHalfLoss(sfm.LossFunction):
    # Scaling
    # -------
    # Given one robustifier
    #   s -> rho(s)
    # one can change the length scale at which robustification takes
    # place, by adding a scale factor 'a' as follows:
    #
    #   s -> a^2 rho(s / a^2).
    #
    # The first and second derivatives are:
    #
    #   s -> rho'(s / a^2),
    #   s -> (1 / a^2) rho''(s / a^2),
    
    # rho(s) = 2*|s|^0.5
    # rho'(s) = sign(s)|s|^(-0.5)
    # rho''(s) = |s|^(-1.5)
    
    def __init__(self,a):
        self.a = a
        self.sqrt_a = sqrt(a)
        super().__init__()
    def Evaluate(self, sq_norm, out):
        out[0] = 2.0*self.a*self.sqrt_a*pow(sq_norm,0.25)
        if sq_norm < 0.01:
            sq_norm = 0.01
        out[1] = 0.5*pow(self.a,-1.5)*pow(sq_norm,-0.75)
        out[2] = -0.375*self.a*self.sqrt_a*pow(sq_norm,-1.75)
class LTwoLoss(sfm.LossFunction):
    # Scaling
    # -------
    # Given one robustifier
    #   s -> rho(s)
    # one can change the length scale at which robustification takes
    # place, by adding a scale factor 'a' as follows:
    #
    #   s -> a^2 rho(s / a^2).
    #
    # The first and second derivatives are:
    #
    #   s -> rho'(s / a^2),
    #   s -> (1 / a^2) rho''(s / a^2),
    
    def __init__(self,a,sigma2):
        self.a_sq = a*a
        super().__init__()
    def Evaluate(self, sq_norm, out):
        out[0] = sq_norm*sq_norm/(self.a_sq*2.0)
        out[1] = sq_norm/self.a_sq
        out[2] = 1/self.a_sq
        
class GemanMcClureLoss(sfm.LossFunction):
    def __init__(self, a, sigma2):
        self.a_sq = a*a
        self.sigma2 = sigma2
        super().__init__()
        
    def Evaluate(self, sq_norm, out):
        out[0] = self.a_sq * self.sigma2*sq_norm / (2.0*(sq_norm+self.a_sq*self.sigma2))
        out[1] = (self.sigma2**2) / (2.0*(sq_norm/self.a_sq + self.sigma2)**2)
        out[2] = -(self.sigma2**2) / (self.a_sq*(sq_norm/self.a_sq+self.sigma2)**3) 

class ComposedLoss(sfm.LossFunction):
    #   rho(s) = f(g(s))
    def __init__(self,f:sfm.LossFunction, g:sfm.LossFunction):
        self.f = f
        self.g = g
        super().__init__()
    def Evaluate(self, sq_norm, out):
        out_f = [0,0,0]
        out_g = [0,0,0]
        self.g.Evaluate(sq_norm, out_g)
        self.f.Evaluate(out_g[0], out_f)
        out[0] = out_f[0]
        # f'(g(s)) * g'(s).
        out[1] = out_f[1] * out_g[1];
        # f''(g(s)) * g'(s) * g'(s) + f'(g(s)) * g''(s).
        out[2] = out_f[2] * out_g[1] * out_g[1] + out_f[1] * out_g[2]

class ScaledLoss(sfm.LossFunction):
    # If rho is the wrapped robustifier, then this simply outputs
    # s -> a * rho(s)
    # The first and second derivatives are, not surprisingly
    # s -> a * rho'(s)
    # s -> a * rho''(s)
    def __init__(self, rho:sfm.LossFunction, a):
        self.rho = rho
        self.a = a
        super().__init__()
    def Evaluate(self, sq_norm, out):
        self.rho.Evaluate(sq_norm, out)
        out[0] *= self.a
        out[1] *= self.a
        out[2] *= self.a
         
        

class MAGSACWeightBasedLoss(sfm.LossFunction):
    def __init__(self,sigma,inverse = False):
        self.sigma_max = sigma
        self.nu = sfm.nu3
        self.squared_sigma = self.sigma_max * self.sigma_max
        self.squared_sigma_max_2 = 2.0 * self.squared_sigma
        self.cubed_sigma_max = self.squared_sigma*self.sigma_max
        self.dof_minus_one_per_two = (self.nu - 1.0) / 2.0
        self.C_times_two_ad_dof = sfm.C3 * (2**self.dof_minus_one_per_two)
        self.one_over_sigma = self.C_times_two_ad_dof / self.sigma_max
        self.gamma_value = sfm.tgamma(self.dof_minus_one_per_two)
        self.gamma_difference = self.gamma_value - sfm.upper_incomplete_gamma_of_k3
        self.weight_zero = self.one_over_sigma * self.gamma_difference
        self.use_weight_inverse = inverse
        
        super().__init__()
        
    def Evaluate(self, squared_residual, rho):
        
        zero_derivative = False;
        if (squared_residual>sfm.sigma_quantile3*sfm.sigma_quantile3*self.squared_sigma):
            squared_residual = sfm.sigma_quantile3*sfm.sigma_quantile3*self.squared_sigma
            zero_derivative = True
        
        x = round(sfm.precision_of_stored_gamma3 * squared_residual / self.squared_sigma_max_2)
        if sfm.stored_gamma_number3 < x:
            x = sfm.stored_gamma_number3
        s = x * self.squared_sigma_max_2 / sfm.precision_of_stored_gamma3
        
        weight = self.one_over_sigma * (sfm.stored_gamma_values3[x] - sfm.upper_incomplete_gamma_of_k3)
        weight_derivative = -self.C_times_two_ad_dof\
            * ((s/self.squared_sigma_max_2)**(self.nu/2 - 1.5))\
            * exp(-s/self.squared_sigma_max_2) / (2*self.cubed_sigma_max)
        if(s < 1e-7):
            s = 1e-7
        weight_second_derivative = 2.0 * self.C_times_two_ad_dof *\
            ((s/self.squared_sigma_max_2)**(self.nu/2 - 1.5)) *\
            (1.0/self.squared_sigma - (self.nu-3)/s) *\
            exp(-s/self.squared_sigma_max_2) / (8*self.cubed_sigma_max);
        
        if self.use_weight_inverse:
            rho[0] = 1.0/weight;
            rho[1] = -1.0/(weight*weight)*weight_derivative;
            rho[2] = 2.0/(weight*weight*weight)*weight_derivative*weight_derivative\
                - weight_second_derivative/(weight*weight);
            if zero_derivative:
                rho[1] = 0.00001; 
                rho[2] = 0.0;
        else:
            rho[0] = self.weight_zero-weight
            rho[1] = -weight_derivative
            rho[2] = -weight_second_derivative
            if rho[1] == 0:
                rho[1] = 0.00001;
            if zero_derivative:
                rho[1] = 0.00001; 
                rho[2] = 0.0;


class MAGSACWeightBasedLoss4(sfm.LossFunction):
    def __init__(self,sigma,inverse = True):
        self.sigma_max = sigma
        self.nu = sfm.nu4
        self.squared_sigma = self.sigma_max * self.sigma_max
        self.squared_sigma_max_2 = 2.0 * self.squared_sigma
        self.cubed_sigma_max = self.squared_sigma*self.sigma_max
        self.dof_minus_one_per_two = (self.nu - 1.0) / 2.0
        self.C_times_two_ad_dof = sfm.C4 * (2**self.dof_minus_one_per_two)
        self.one_over_sigma = self.C_times_two_ad_dof / self.sigma_max
        self.gamma_value = sfm.tgamma(self.dof_minus_one_per_two)
        self.gamma_difference = self.gamma_value - sfm.upper_incomplete_gamma_of_k4
        self.weight_zero = self.one_over_sigma * self.gamma_difference
        self.use_weight_inverse = inverse
        
        super().__init__()
        
    def Evaluate(self, squared_residual, rho):
        
        zero_derivative = False;
        if (squared_residual>sfm.sigma_quantile4*sfm.sigma_quantile4*self.squared_sigma):
            squared_residual = sfm.sigma_quantile4*sfm.sigma_quantile4*self.squared_sigma
            zero_derivative = True
        
        x = round(sfm.precision_of_stored_gamma4 * squared_residual / self.squared_sigma_max_2)
        if sfm.stored_gamma_number4 < x:
            x = sfm.stored_gamma_number4
        s = x * self.squared_sigma_max_2 / sfm.precision_of_stored_gamma4
        
        weight = self.one_over_sigma * (sfm.stored_gamma_values4[x] - sfm.upper_incomplete_gamma_of_k4)
        weight_derivative = -self.C_times_two_ad_dof\
            * ((s/self.squared_sigma_max_2)**(self.nu/2 - 1.5))\
            * exp(-s/self.squared_sigma_max_2) / (2*self.cubed_sigma_max)
        if(s < 1e-7):
            s = 1e-7
        weight_second_derivative = 2.0 * self.C_times_two_ad_dof *\
            ((s/self.squared_sigma_max_2)**(self.nu/2 - 1.5)) *\
            (1.0/self.squared_sigma - (self.nu-3)/s) *\
            exp(-s/self.squared_sigma_max_2) / (8*self.cubed_sigma_max);
        
        if self.use_weight_inverse:
            rho[0] = 1.0/weight;
            rho[1] = -1.0/(weight*weight)*weight_derivative;
            rho[2] = 2.0/(weight*weight*weight)*weight_derivative*weight_derivative\
                - weight_second_derivative/(weight*weight);
            if zero_derivative:
                rho[1] = 0.00001; 
                rho[2] = 0.0;
        else:
            rho[0] = self.weight_zero-weight
            rho[1] = -weight_derivative
            rho[2] = -weight_second_derivative
            if rho[1] == 0:
                rho[1] = 0.00001;
            if zero_derivative:
                rho[1] = 0.00001; 
                rho[2] = 0.0;

class MAGSACWeightBasedLoss9(sfm.LossFunction):
    def __init__(self,sigma,inverse = False):
        self.sigma_max = sigma
        self.nu = sfm.nu9
        self.squared_sigma = self.sigma_max * self.sigma_max
        self.squared_sigma_max_2 = 2.0 * self.squared_sigma
        self.cubed_sigma_max = self.squared_sigma*self.sigma_max
        self.dof_minus_one_per_two = (self.nu - 1.0) / 2.0
        self.C_times_two_ad_dof = sfm.C9 * (2**self.dof_minus_one_per_two)
        self.one_over_sigma = self.C_times_two_ad_dof / self.sigma_max
        self.gamma_value = sfm.tgamma(self.dof_minus_one_per_two)
        self.gamma_difference = self.gamma_value - sfm.upper_incomplete_gamma_of_k9
        self.weight_zero = self.one_over_sigma * self.gamma_difference
        self.use_weight_inverse = inverse
        
        super().__init__()
        
    def Evaluate(self, squared_residual, rho):
        
        zero_derivative = False;
        if (squared_residual>sfm.sigma_quantile9*sfm.sigma_quantile9*self.squared_sigma):
            squared_residual = sfm.sigma_quantile9*sfm.sigma_quantile9*self.squared_sigma
            zero_derivative = True
        
        x = round(sfm.precision_of_stored_gamma9 * squared_residual / self.squared_sigma_max_2)
        if sfm.stored_gamma_number9 < x:
            x = sfm.stored_gamma_number9
        s = x * self.squared_sigma_max_2 / sfm.precision_of_stored_gamma9
        
        weight = self.one_over_sigma * (sfm.stored_gamma_values9[x] - sfm.upper_incomplete_gamma_of_k9)
        weight_derivative = -self.C_times_two_ad_dof\
            * ((s/self.squared_sigma_max_2)**(self.nu/2 - 1.5))\
            * exp(-s/self.squared_sigma_max_2) / (2*self.cubed_sigma_max)
        if(s < 1e-7):
            s = 1e-7
        weight_second_derivative = 2.0 * self.C_times_two_ad_dof *\
            ((s/self.squared_sigma_max_2)**(self.nu/2 - 1.5)) *\
            (1.0/self.squared_sigma - (self.nu-3)/s) *\
            exp(-s/self.squared_sigma_max_2) / (8*self.cubed_sigma_max);
        
        if self.use_weight_inverse:
            rho[0] = 1.0/weight;
            rho[1] = -1.0/(weight*weight)*weight_derivative;
            rho[2] = 2.0/(weight*weight*weight)*weight_derivative*weight_derivative\
                - weight_second_derivative/(weight*weight);
            if zero_derivative:
                rho[1] = 0.00001; 
                rho[2] = 0.0;
        else:
            rho[0] = self.weight_zero-weight
            rho[1] = -weight_derivative
            rho[2] = -weight_second_derivative
            
            if rho[1] == 0:
                rho[1] = 0.00001;
            if zero_derivative:
                rho[1] = 0.00001; 
                rho[2] = 0.0;