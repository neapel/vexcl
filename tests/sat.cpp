#define BOOST_TEST_MODULE SummedAreaTable
#include <vexcl/sat.hpp>
#include <boost/test/unit_test.hpp>
#include "context_setup.hpp"
#include <boost/test/test_case_template.hpp>
#include <boost/mpl/list.hpp>

typedef boost::mpl::list<int, double> test_types;

BOOST_AUTO_TEST_CASE_TEMPLATE(correct_sat, T, test_types)
{
    std::vector<cl::CommandQueue> queue(1, ctx.queue(0));

    for(size_t i = 0 ; i < 20 ; i++) {
        size_t w = 1 + rand() % 4096, h = 1 + rand() % 4096;
        if(rand() % 4 == 0) w -= w % 32;
        if(rand() % 4 == 0) h -= h % 32;

        std::vector<T> in = random_vector<T>(w * h), ref(w * h), out(w * h);

        // compute reference on CPU.
        for(size_t y = 0 ; y < h ; y++)
            for(size_t x = 0 ; x < w ; x++)
                ref[x + y * w] = in[x + y * w]
                    + (y > 0 ? ref[x + (y - 1) * w] : 0)
                    + (x > 0 ? ref[x - 1 + y * w] : 0)
                    - (y > 0 && x > 0 ? ref[x - 1 + (y - 1) * w] : 0);

        vex::SAT<T> sat(queue, w, h);
        vex::vector<T> d_in(queue, in);
        d_in = sat(d_in);
        vex::copy(d_in, out);

        double s = 0;
        for(size_t i = 0 ; i < w * h ; i++)
            s += std::abs(out[i] - ref[i]);
        s /= (w * h);

        BOOST_CHECK_SMALL(s, 1e-6);
    }
}

BOOST_AUTO_TEST_SUITE_END()
