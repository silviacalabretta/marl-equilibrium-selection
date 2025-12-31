import argparse

from src.game import game_dictionary
from src.learning_rule import learning_rule_dictionary
from src.unified_learning import UnifiedLearning


def parse_args():
    parser = argparse.ArgumentParser(description="Run an Equilibrium Selection Learning Framework")

    parser.add_argument("--iterations", type=int, default=1000, help="Learning iterations")
    parser.add_argument(
        "--game", 
        type=str, 
        default="treasure", 
        choices=["treasure","staghunt"],
        help = "Game to play: treasure or staghunt"
    )
    parser.add_argument(
        "--learning-rule",
        type=str,
        default="loglinear",
        choices=["loglinear", "mardenmood"],
        help="Learning rule: loglinear or mardenmood",
    )
    parser.add_argument(
        "--rule-coeffs",
        type=float,
        nargs="+",
        default=[0.01],
        help="Coefficients for the learning rule ([eps] for LogLinear, [eps,c] for MardenMood)",
    )
    parser.add_argument("--save", action="store_true", default=False, 
                        help="If present, plots will be saved instead of being shown. Use --output-path to specify a path")
    
    parser.add_argument("--output-path", type=str, default=None,
                        help="Saving path for the produced plot (e.g. \"out/plot.png\")")
    
    parser.add_argument("--no-override", action="store_true", default=False, 
                        help="If present, plot functions will produce different filenames to avoid ovverriding.")
    
    parser.add_argument("--num-runs", type=int, default=1, 
                        help="Number of independent learning trajectories to run. ")

    return parser.parse_args()


def objects_setup(args):

    GameClass = game_dictionary.get(args.game)
    if not GameClass:
        raise ValueError(f"Game not valid: {args.game}")
        
    game = GameClass()


    RuleClass = learning_rule_dictionary.get(args.learning_rule)
    if not RuleClass:
        raise ValueError(f"Learning rule not valid: {args.learning_rule}")

    if args.learning_rule == "loglinear":
        if len(args.rule_coeffs) != 1:
            raise ValueError(
                f"LogLinear rule requires exactly 1 coefficient, received: {len(args.rule_coeffs)}"
            )
        epsilon = args.rule_coeffs[0]
        learning_rule = RuleClass(epsilon=epsilon)
    
    elif args.learning_rule == "mardenmood":
        if len(args.rule_coeffs) != 2:
            raise ValueError(
                f"Marden Mood rule requires exactly 2 coefficients, received: {len(args.rule_coeffs)}"
            )
        epsilon = args.rule_coeffs[0]
        c = args.rule_coeffs[1]
        learning_rule = RuleClass(epsilon=epsilon, c=c)
    else:
        raise NotImplementedError(f"Missing logic for rule: {args.learning_rule}")
        
    return game, learning_rule
    

def main():
    args = parse_args()

    game, learning_rule = objects_setup(args)

    learner = UnifiedLearning(game=game, T=args.iterations, learning_rule=learning_rule)
    learner.run()

    # Print the final V-values and Q-values learnt by player 0
    learner.print_results()

    # Plot the evolution of the V-value of player 0 in the initial state 
    learner.plot_convergence(save=args.save, save_path=args.output_path, no_override=args.no_override)

    # Plot the evolution of the policy in the initial state (just actions (0,0) and (1,1))
    learner.plot_policy_evolution(learner.s1_action_history, params=args.rule_coeffs, save=args.save, save_path=args.output_path, no_override=args.no_override)
    
    if args.num_runs > 1:
        actions = learner.run_simulations(num_runs=args.num_runs)
        learner.plot_policy_evolution(actions, params=args.rule_coeffs, save=args.save, save_path=args.output_path, no_override=args.no_override)


if __name__ == "__main__":
    main()