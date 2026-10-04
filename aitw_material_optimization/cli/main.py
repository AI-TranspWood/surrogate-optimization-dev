import click


@click.command()
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_spec_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('result_file', type=click.Path(writable=True, dir_okay=False))
@click.option('--mode', type=click.Choice(['matching', 'constrained']), required=True)
@click.option('--maxiter', type=click.IntRange(min=0), default=None, help='Maximum generations (matching: 20; constrained: 10).')
@click.option('--popsize', type=click.IntRange(min=1), default=None, help='Population multiplier per free variable (matching: 200; constrained: 1000).')
@click.option('--tol', type=click.FloatRange(min=0), default=1e-4, show_default=True)
@click.option('--seed', type=click.IntRange(min=0, max=2**32 - 1), default=None)
def aitw_material_optimization(input_file, output_spec_file, result_file, mode, maxiter, popsize, tol, seed):
    """Optimize a material using input and stiffness specifications in two JSON files."""
    from aitw_material_optimization import optimize
    from aitw_material_optimization import dataLoaders as dl

    settings = {'tol': tol, 'seed': seed}
    if maxiter is not None:
        settings['maxiter'] = maxiter
    if popsize is not None:
        settings['popsize'] = popsize
    try:
        result = optimize(dl.loadInput(input_file), dl.loadInput(output_spec_file), mode, settings)
        dl.saveResult(result, result_file)
    except (ValueError, OSError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f'Optimization result saved to {result_file} ({result["optimizer"]["status"]})')
    if not result['feasible']:
        raise click.ClickException('The returned candidate is infeasible; see the result file for details.')
