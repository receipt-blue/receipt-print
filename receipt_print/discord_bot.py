import asyncio
import io  # For reading image bytes
import math
import os
import re
import subprocess
from pathlib import Path

import discord
from discord.ext import commands

# --- Bot Configuration ---
from dotenv import load_dotenv
from PIL import Image  # For image dimension checking

load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DEFAULT_FOOTER_FROM_ENV = os.getenv("DEFAULT_FOOTER")

ALLOWED_CHANNEL_IDS = {
    int(cid) for cid in os.getenv("ALLOWED_CHANNEL_IDS", "").split(",") if cid
}
# if ALLOWED_CHANNEL_IDS is empty, bot will respond in any channel it's in.

intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
bot = commands.Bot(command_prefix="!rp-", intents=intents)

# --- Constants ---
EMOJI_PRINTER = "🖨️"
EMOJI_CHECK = "✅"
EMOJI_CROSS = "❌"
EMOJI_WARNING = "⚠️"

CACHE_BASE_DIR = Path.home() / ".local" / "share" / "receipt-print" / "discord"
BOT_MAX_LINES = 100
RP_CHAR_WIDTH = int(os.getenv("RP_CHAR_WIDTH", "42"))
DOTS_PER_LINE = 24
RP_PRINTER_MAX_IMG_WIDTH = int(os.getenv("RP_PRINTER_MAX_IMG_WIDTH", "576"))


# --- Helper Functions ---
def count_text_lines(text: str, width: int) -> int:
    """Counts how many printed lines (with wrapping) a given text will produce."""
    total = 0
    if not text:
        return 0
    for ln in text.splitlines():
        if not ln:  # Empty line itself counts as one line
            total += 1
        else:
            total += (len(ln) + width - 1) // width
    return total


async def get_image_dimensions(attachment: discord.Attachment):
    """Gets dimensions of an image attachment."""
    try:
        image_bytes = await attachment.read()
        img = Image.open(io.BytesIO(image_bytes))
        return img.size  # (width, height)
    except Exception as e:
        print(f"Error getting image dimensions for {attachment.filename}: {e}")
        return None


async def estimate_image_lines(
    attachment: discord.Attachment,
    user_scale_str: str,
    user_align_str: str,
    printer_max_width: int = RP_PRINTER_MAX_IMG_WIDTH,
    dots_per_line: int = DOTS_PER_LINE,
):
    """Estimates printable lines for an image, considering scaling and orientation."""
    original_dims = await get_image_dimensions(attachment)
    if not original_dims:
        return 0

    original_width, original_height = float(original_dims[0]), float(original_dims[1])
    if original_width <= 0 or original_height <= 0:  # Image has no printable area
        return 0

    scaled_width, scaled_height = original_width, original_height
    if original_width > printer_max_width:
        ratio = float(printer_max_width) / original_width
        scaled_width = float(printer_max_width)
        scaled_height = original_height * ratio

    user_scale = 1.0
    try:
        if user_scale_str:  # Ensure string is not empty before trying to float
            parsed_scale = float(user_scale_str)
            user_scale = parsed_scale  # Allow any float value for estimation
    except ValueError:
        pass  # Keep default 1.0 if parsing fails

    final_scaled_width = scaled_width * user_scale
    final_scaled_height = scaled_height * user_scale

    is_landscape = user_align_str.lower() in {"l-top", "l-bottom", "l-center"}
    height_for_printing = final_scaled_width if is_landscape else final_scaled_height

    if height_for_printing < 0:  # If scale was negative
        height_for_printing = 0.0
    return math.ceil(height_for_printing / dots_per_line)


def parse_parameters(content: str) -> dict:
    """
    Parses !keyword value pairs from content.
    Values are taken as the string between one !keyword and the next (or end of content).
    """
    params = {}
    # split by !keyword, keeping the keyword as a delimiter in the resulting list
    segments = re.split(r"(!\w+)", content)

    i = 0
    while i < len(segments):
        segment = segments[i].strip()
        if segment.startswith("!"):  # this is a keyword segment
            key = segment
            value_candidate = ""
            # the value is in the *next* segment, if that segment exists,
            # is not empty after stripping, and is not another keyword.
            if (
                (i + 1) < len(segments)
                and segments[i + 1].strip()
                and not segments[i + 1].strip().startswith("!")
            ):
                value_candidate = segments[i + 1].strip()
                i += 1  # Consume the value segment as it's now processed along with the key

            # handle flags
            if key in ["!debug", "!override-length"]:
                params[key] = True
            # handle parameters that expect a value
            elif value_candidate:  # If a non-empty value string was found
                params[key] = value_candidate
            # handle parameters that can be explicitly empty
            elif key in ["!caption", "!footer"]:
                if not value_candidate and key in ["!caption", "!footer"]:
                    params[key] = ""  # explicitly set as empty string
        i += 1
    return params


async def run_receipt_print_command(
    command_args, stdin_text=None, suppress_line_limit_env=False
):
    """Runs a receipt-print command as a subprocess."""
    env = os.environ.copy()
    if suppress_line_limit_env:
        env["RP_MAX_LINES"] = "999999"
    else:
        env["RP_MAX_LINES"] = str(BOT_MAX_LINES + 10)

    try:
        process = await asyncio.create_subprocess_exec(
            "receipt-print",
            *command_args,
            stdin=subprocess.PIPE if stdin_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await process.communicate(
            input=stdin_text.encode("utf-8") if stdin_text is not None else None
        )
        return stdout.decode("utf-8"), stderr.decode("utf-8"), process.returncode
    except FileNotFoundError:
        return (
            "",
            "Error: `receipt-print` command not found. Is it installed and in PATH?",
            -1,
        )
    except Exception as e:
        return "", f"Error running subprocess: {e}", -1


# --- Event Handlers ---
@bot.event
async def on_ready():
    print(f"Bot '{bot.user.name}' is online. Current time: {discord.utils.utcnow()}")
    if DEFAULT_FOOTER_FROM_ENV:
        print(
            f'Default footer from .env (used if no !caption and no user-defined !footer): "{DEFAULT_FOOTER_FROM_ENV}"'
        )
    if not ALLOWED_CHANNEL_IDS:
        print(
            "Warning: No ALLOWED_CHANNEL_IDS set. Bot will respond in all channels it has access to."
        )
    else:
        print(f"Monitoring channels: {ALLOWED_CHANNEL_IDS}")


@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return
    if ALLOWED_CHANNEL_IDS and message.channel.id not in ALLOWED_CHANNEL_IDS:
        return

    has_images = any(
        att.content_type and att.content_type.startswith("image/")
        for att in message.attachments
    )
    has_code_blocks = (
        re.search(r"```(?:[a-zA-Z0-9_+\-]*\n)?([\s\S]+?)```", message.content)
        is not None
    )

    if has_images or has_code_blocks:
        try:
            await message.add_reaction(EMOJI_PRINTER)
        except discord.Forbidden:
            print(
                f"Could not add reaction in {message.channel.id}. Missing 'Add Reactions' permission."
            )
        except Exception as e:
            print(f"Error adding reaction: {e}")

    await bot.process_commands(message)


@bot.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    if payload.user_id == bot.user.id or str(payload.emoji) != EMOJI_PRINTER:
        return

    channel = bot.get_channel(payload.channel_id)
    if not channel or (ALLOWED_CHANNEL_IDS and channel.id not in ALLOWED_CHANNEL_IDS):
        return

    try:
        message = await channel.fetch_message(payload.message_id)
    except (discord.NotFound, discord.Forbidden):
        return

    for reaction in message.reactions:
        if reaction.me and str(reaction.emoji) in [
            EMOJI_CHECK,
            EMOJI_CROSS,
            EMOJI_WARNING,
        ]:
            try:
                await message.remove_reaction(reaction.emoji, bot.user)
            except Exception:
                pass

    params = parse_parameters(message.content)
    image_attachments = [
        att
        for att in message.attachments
        if att.content_type and att.content_type.startswith("image/")
    ]
    code_block_contents = [
        match.group(1).strip()
        for match in re.finditer(
            r"```(?:[a-zA-Z0-9_+\-]*\n)?([\s\S]+?)```", message.content
        )
    ]

    final_code_text_to_print = None
    if code_block_contents:
        final_code_text_to_print = (
            "\n\n---\n\n".join(code_block_contents)
            if len(code_block_contents) > 1
            else code_block_contents[0]
        )

    if not image_attachments and not final_code_text_to_print:
        await message.add_reaction(EMOJI_CROSS)
        return

    total_estimated_lines = 0
    if final_code_text_to_print:
        total_estimated_lines += count_text_lines(
            final_code_text_to_print, RP_CHAR_WIDTH
        )

    scale_values_str_list = params.get("!scale", "1.0").split(",")
    align_values_str_list = params.get("!align", "center").split(",")

    if image_attachments:
        for i, att in enumerate(image_attachments):
            current_scale_str = (
                scale_values_str_list[i]
                if i < len(scale_values_str_list)
                else scale_values_str_list[-1]
            ).strip()
            current_align_str = (
                align_values_str_list[i]
                if i < len(align_values_str_list)
                else align_values_str_list[-1]
            ).strip()
            img_lines = await estimate_image_lines(
                att, current_scale_str, current_align_str
            )
            total_estimated_lines += img_lines

    override_length_flag = params.get("!override-length", False)
    should_suppress_rp_limit_env = override_length_flag or (
        total_estimated_lines <= BOT_MAX_LINES
    )

    if total_estimated_lines > BOT_MAX_LINES and not override_length_flag:
        await message.add_reaction(EMOJI_WARNING)
        await channel.send(
            f"Estimated print is {total_estimated_lines} lines (limit: {BOT_MAX_LINES}). "
            f"Add `!override-length` to your message and click {EMOJI_PRINTER} again to print.",
            reference=message,
            delete_after=60,
        )
        return

    print_success_overall = True
    error_messages_list = []

    if final_code_text_to_print:
        stdout, stderr, rc = await run_receipt_print_command(
            [],
            stdin_text=final_code_text_to_print,
            suppress_line_limit_env=should_suppress_rp_limit_env,
        )
        if rc != 0:
            print_success_overall = False
            error_messages_list.append(
                f"Error printing text:\n```\n{stderr or stdout or 'Unknown error'}\n```"
            )

    if image_attachments and print_success_overall:
        message_cache_dir = CACHE_BASE_DIR / str(channel.id) / str(message.id)
        message_cache_dir.mkdir(parents=True, exist_ok=True)

        image_paths_to_print = []
        for i, att in enumerate(image_attachments):
            safe_filename = f"{i}_{re.sub(r'[^\w.-]', '_', att.filename)}"
            local_image_path = message_cache_dir / safe_filename
            try:
                await att.save(local_image_path)
                image_paths_to_print.append(str(local_image_path))
            except Exception as e:
                print_success_overall = False
                error_messages_list.append(f"Error saving image {att.filename}: {e}")
                break

        if image_paths_to_print and print_success_overall:
            img_cmd_args = ["image"]

            # caption logic (only from !caption)
            user_provided_caption_str = params.get("!caption")
            if user_provided_caption_str is not None:  # Allows !caption "" to pass ""
                img_cmd_args.extend(["--caption", user_provided_caption_str])

            # footer logic: only applies if no !caption was provided by the user
            footer_to_pass = None
            if user_provided_caption_str is None:
                user_defined_footer = params.get("!footer")
                if user_defined_footer is not None:  # user explicitly used !footer
                    # we use it even if it's "", CLI can decide if "" is a no-op
                    footer_to_pass = user_defined_footer
                elif (
                    DEFAULT_FOOTER_FROM_ENV
                ):  # no user !footer, but default from .env exists
                    footer_to_pass = DEFAULT_FOOTER_FROM_ENV

            if (
                footer_to_pass is not None
            ):  # pass if footer_to_pass is set (could be "" from user)
                img_cmd_args.extend(["--footer", footer_to_pass])

            # other image parameters
            if "!scale" in params:
                img_cmd_args.extend(["--scale", params["!scale"]])
            if "!align" in params:
                img_cmd_args.extend(["--align", params["!align"]])
            if "!dither" in params:
                img_cmd_args.extend(["--dither", params["!dither"]])
            if "!threshold" in params:
                img_cmd_args.extend(["--threshold", params["!threshold"]])
            if "!diffusion" in params:
                img_cmd_args.extend(["--diffusion", params["!diffusion"]])
            if "!spacing" in params:
                img_cmd_args.extend(["--spacing", params["!spacing"]])
            if params.get("!debug", False):
                img_cmd_args.append("--debug")

            img_cmd_args.extend(image_paths_to_print)

            stdout, stderr, rc = await run_receipt_print_command(
                img_cmd_args, suppress_line_limit_env=should_suppress_rp_limit_env
            )
            if rc != 0:
                print_success_overall = False
                error_messages_list.append(
                    f"Error printing images:\n```\n{stderr or stdout or 'Unknown error'}\n```"
                )

    if print_success_overall:
        await message.add_reaction(EMOJI_CHECK)
        for reaction in message.reactions:
            if str(reaction.emoji) == EMOJI_WARNING and reaction.me:
                try:
                    await message.remove_reaction(EMOJI_WARNING, bot.user)
                except Exception:
                    pass
    else:
        await message.add_reaction(EMOJI_CROSS)
        full_error_report = (
            "\n".join(error_messages_list)
            if error_messages_list
            else "An unspecified error occurred."
        )
        if len(full_error_report) > 1900:
            full_error_report = full_error_report[:1900] + "..."
        await channel.send(
            f"Printing failed:\n{full_error_report}",
            reference=message,
            delete_after=120,
        )


# --- Bot Startup ---
if __name__ == "__main__":
    if not TOKEN:
        print("Error: DISCORD_BOT_TOKEN environment variable not found.")
    else:
        if not ALLOWED_CHANNEL_IDS:
            print(
                "Warning: ALLOWED_CHANNEL_IDS is not set or empty. The bot will respond in all channels it has access to."
            )
            print(
                "Consider setting the environment variable (e.g., 'ALLOWED_CHANNEL_IDS=123,456') to restrict channels."
            )
        try:
            bot.run(TOKEN)
        except discord.PrivilegedIntentsRequired:
            print(
                "Error: Privileged intents (Message Content, Reactions) are not enabled for the bot in the Discord Developer Portal."
            )
        except Exception as e:
            print(f"Error running bot: {e}")
