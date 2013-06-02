#include <iostream>
#include <vector>
#include <utility>
#include <tuple>
#include <random>
#include <algorithm>

#include <mpi.h>

#include <vexcl/vexcl.hpp>
#include <vexcl/mpi.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/algebra/vector_space_algebra.hpp>

/* Resizing of vex::mpi types for odeint */
namespace boost { namespace numeric { namespace odeint {

// vex::mpi::multivector
template< typename T , size_t N, bool own >
struct is_resizeable< vex::mpi::multivector< T , N , own > > : boost::true_type { };

template< typename T , size_t N, bool own >
struct resize_impl< vex::mpi::multivector< T , N , own > , vex::mpi::multivector< T , N , own > >
{
    static void resize( vex::mpi::multivector< T , N , own > &x1 , const vex::mpi::multivector< T , N , own > &x2 )
    {
        x1.resize( x2 );
    }
};

template< typename T , size_t N, bool own >
struct same_size_impl< vex::mpi::multivector< T , N , own > , vex::mpi::multivector< T , N , own > >
{
    static bool same_size( const vex::mpi::multivector< T , N , own > &x1 , const vex::mpi::multivector< T , N , own > &x2 )
    {
        return x1.local_size() == x2.local_size();
    }
};

} } }

namespace odeint = boost::numeric::odeint;

typedef double value_type;
typedef vex::mpi::vector< value_type >         vector_type;
typedef vex::mpi::multivector< value_type, 3 > state_type;

const static value_type sigma = 10.0;
const static value_type b     = 8.0 / 3.0;

struct lorenz_system {
    const vector_type &R;

    lorenz_system( const vector_type &R ) : R( R ) { }

    void operator()( const state_type &x, state_type &dxdt, value_type ) {
	dxdt = std::tie(
		sigma * (x(1) - x(0)),
		R * x(0) - x(1) - x(0) * x(2),
		x(0) * x(1) - b * x(2)
		);
    }
};

int main( int argc , char **argv ) {
    const size_t n = argc > 1 ? atoi(argv[1]) : 1024;

    const value_type dt    = 0.01;
    const value_type t_max = 100.0;
    const value_type Rmin  = 0.1;
    const value_type Rmax  = 50.0;
    const value_type dR    = (Rmax - Rmin) / (n - 1);


    MPI_Init(&argc, &argv);
    vex::mpi::comm_data mpi(MPI_COMM_WORLD);

    size_t chunk_size = (n + mpi.size - 1) / mpi.size;
    size_t chunk_start = mpi.rank * chunk_size;
    size_t chunk_end   = std::min(n, chunk_start + chunk_size);
    chunk_size = chunk_end - chunk_start;

    auto part = mpi.restore_partitioning(chunk_size);

    try {
        vex::Context ctx( vex::Filter::Exclusive(
                    vex::Filter::Env && vex::Filter::Count(1) ) );

        mpi.precondition(!ctx.empty(), "No OpenCL devices found");

        for(int i = 0; i < mpi.size; ++i) {
            if (i == mpi.rank)
                std::cout << mpi.rank << ": " << ctx.device(0) << std::endl;

            MPI_Barrier(mpi.comm);
        }

        state_type  X(mpi.comm, ctx, chunk_size);
        vector_type R(mpi.comm, ctx, chunk_size);

        X = 10.0;
        R = Rmin + dR * vex::element_index(part[mpi.rank]);

        odeint::runge_kutta4<
            state_type, value_type, state_type, value_type,
            odeint::vector_space_algebra, odeint::default_operations
            > stepper;

        odeint::integrate_const(stepper, lorenz_system(R), X, value_type(0.0), t_max, dt);

        std::cout << mpi.rank << ": " << X(0)[0] << std::endl;

    } catch(const cl::Error &e) {
	std::cout << e << std::endl;
    } catch(const std::exception &e) {
	std::cout << e.what() << std::endl;
    }

    MPI_Finalize();
}
