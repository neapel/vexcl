#define BOOST_TEST_MODULE KernelGenerator
#include <boost/test/unit_test.hpp>
#include "context_setup.hpp"

using namespace vex;

template <class state_type>
state_type sys_func(const state_type &x) {
    return sin(x);
}

template <class state_type, class SysFunction>
void runge_kutta_2(SysFunction sys, state_type &x, double dt) {
    state_type k1 = dt * sys(x);
    state_type k2 = dt * sys(x + 0.5 * k1);

    x += k2;
}

BOOST_AUTO_TEST_CASE(kernel_generator)
{
    typedef vex::generator::symbolic<double> sym_state;

    const size_t n  = 1024;
    const double dt = 0.01;

    std::ostringstream body;
    vex::generator::set_recorder(body);

    sym_state sym_x(sym_state::VectorParameter);

    // Record expression sequence.
    runge_kutta_2(sys_func<sym_state>, sym_x, dt);

    // Build kernel.
    auto kernel = vex::generator::build_kernel(
            ctx, "rk2_stepper", body.str(), sym_x);

    std::vector<double> x = random_vector<double>(n);
    vex::vector<double> X(ctx, x);

    for(int i = 0; i < 100; i++) kernel(X);

    check_sample(X, [&](size_t idx, double a) {
            double s = x[idx];
            for(int i = 0; i < 100; i++)
                runge_kutta_2(sys_func<double>, s, dt);

            BOOST_CHECK_CLOSE(a, s, 1e-8);
            });
}

/*
An alternative variant, which does not use the generator facility.
Intermediate subexpression are captured with help of 'auto' keyword, and
are combined into larger expression.

This is not as effective as generated kernel, because same input vector
(here 'x') is passed as several different parameters. This specific example
takes about twice as long to execute as the above variant.

Nevertheless, this may be more convenient in some cases.
*/
BOOST_AUTO_TEST_CASE(lazy_evaluation)
{
    const size_t n  = 1024;
    const double dt = 0.01;

    auto rk2 = [](vex::vector<double> &x, double dt) {
        auto k1 = dt * sin(x);
        auto x1 = x + 0.5 * k1;

        auto k2 = dt * sin(x1);

        x += k2;
    };

    std::vector<double> x = random_vector<double>(n);
    vex::vector<double> X(ctx, x);

    for(int i = 0; i < 100; i++) {
        rk2(X, dt);
        // Temporary workaround for ati bug:
        // http://devgurus.amd.com/message/1295503#1295503
        for(unsigned d = 0; d < ctx.size(); ++d)
            ctx.queue(d).finish();
    }

    check_sample(X, [&](size_t idx, double a) {
            double s = x[idx];
            for(int i = 0; i < 100; i++)
                runge_kutta_2(sys_func<double>, s, dt);

            BOOST_CHECK_CLOSE(a, s, 1e-8);
            });
}

BOOST_AUTO_TEST_SUITE_END()
