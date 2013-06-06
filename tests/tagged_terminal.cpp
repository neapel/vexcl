#define BOOST_TEST_MODULE TaggedTerminal
#include <boost/test/unit_test.hpp>
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(tagged_terminal)
{
    using vex::tag;
    using vex::range;

    const size_t n = 1024;

    std::vector<double> x = random_vector<double>(n);

    vex::vector<double> X(ctx, x);
    vex::vector<double> Y = tag<1>(X) * tag<1>(X);

    check_sample(Y, [&](size_t idx, double v) {
            BOOST_CHECK_CLOSE(v, x[idx] * x[idx], 1e-8); });

    vex::Reductor<double, vex::SUM> sum(ctx);

    BOOST_CHECK_CLOSE(
            sum(tag<3>(2) * tag<1>(X) * tag<1>(X) + tag<3>(2) * tag<2>(Y) * tag<2>(Y)),
            std::accumulate(x.begin(), x.end(), 0.0, [](double sum, double v) {
                double y = v * v;
                return sum + 2 * (y + y * y);
                }),
            1e-6
            );

    vex::slicer<1> slice(&n);

    Y = tag<1>(41) * tag<2>(X) + tag<3>(slice[range()](X));

    check_sample(Y, [&](size_t idx, double v) {
            BOOST_CHECK_CLOSE(v, 42 * x[idx], 1e-8); });
}

BOOST_AUTO_TEST_SUITE_END()
