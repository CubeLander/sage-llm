# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
'''The CLI entrypoints of vLLM

Note that all future modules must be lazily loaded within main
to avoid certain eager import breakage.'''
from __future__ import annotations

import importlib.metadata


def main():
    import vllm.cli.benchmark.main
    import vllm.cli.collect_env
    import vllm.cli.openai
    import vllm.cli.run_batch
    import vllm.cli.serve
    from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup
    from vllm.utils import FlexibleArgumentParser

    CMD_MODULES = [
        vllm.cli.openai,
        vllm.cli.serve,
        vllm.cli.benchmark.main,
        vllm.cli.collect_env,
        vllm.cli.run_batch,
    ]

    cli_env_setup()

    parser = FlexibleArgumentParser(
        description="vLLM CLI",
        epilog=VLLM_SUBCMD_PARSER_EPILOG,
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=importlib.metadata.version('vllm'),
    )
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}
    for cmd_module in CMD_MODULES:
        new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(
                dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
    args = parser.parse_args()
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
