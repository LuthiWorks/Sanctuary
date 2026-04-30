"""World manipulation tools — the entity reshapes its sandbox.

These tools let the entity spawn, move, rotate, resize, recolor, delete,
and inspect objects in the Godot SanctuaryWorld sandbox. Each tool builds
a JSON ``world_command`` and dispatches it through the active
:class:`SanctuaryWebServer`'s ``/ws/world`` channel.

By default tools fire-and-forget: the call returns immediately with an
acknowledgement string and the entity perceives the result via the next
cycle's ``scene_state`` percept. ``get_scene_state`` is the exception —
it awaits the matching ``command_result`` because the tool's return
value *is* the scene state.

Tools are registered on the runner's tool registry by the WebSocket
server when it starts (so dispatch always has a live channel to send
through). Wiring lives in :func:`register_world_tools` below.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sanctuary.tools.registry import ToolRegistry, ToolResult

if TYPE_CHECKING:
    from sanctuary.api.ws_server import SanctuaryWebServer

logger = logging.getLogger(__name__)


PRIMITIVE_TYPES = ("cube", "sphere", "cylinder", "ramp", "plane")
SURFACE_TYPES = ("floor", "wall", "platform")


def _new_command_id() -> str:
    return f"cmd_{uuid4().hex[:12]}"


def _new_object_id() -> str:
    return f"obj_{uuid4().hex[:12]}"


def _make_command(action: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "world_command",
        "command_id": _new_command_id(),
        "action": action,
        "params": params,
    }


def register_world_tools(
    registry: ToolRegistry,
    server: "SanctuaryWebServer",
) -> None:
    """Register the eight world-manipulation tools on ``registry``.

    Each tool is a closure that captures ``server`` so it can dispatch
    commands without a global lookup. Called from
    :meth:`SanctuaryWebServer.start` once the WebSocket channel is live.
    """

    # ------------------------------------------------------------------
    # Spawn / shape tools
    # ------------------------------------------------------------------

    async def _spawn_object(params: dict[str, Any]) -> ToolResult:
        object_type = str(params.get("type", "cube"))
        if object_type not in PRIMITIVE_TYPES:
            return ToolResult(
                tool_name="spawn_object",
                success=False,
                error=f"Unknown type: {object_type!r} (expected one of {PRIMITIVE_TYPES})",
            )
        object_id = _new_object_id()
        cmd_params = {
            "object_id": object_id,
            "object_type": object_type,
            "position": list(params.get("position", [0.0, 0.5, 0.0])),
            "scale": list(params.get("scale", [1.0, 1.0, 1.0])),
            "color": list(params.get("color", [0.7, 0.7, 0.7])),
            "name": str(params.get("name", object_type)),
        }
        await server._send_world_command(_make_command("spawn", cmd_params))
        return ToolResult(
            tool_name="spawn_object",
            success=True,
            output=f"Spawn requested: {object_id} ({object_type})",
        )

    async def _create_surface(params: dict[str, Any]) -> ToolResult:
        surface_type = str(params.get("type", "floor"))
        if surface_type not in SURFACE_TYPES:
            return ToolResult(
                tool_name="create_surface",
                success=False,
                error=f"Unknown surface type: {surface_type!r} (expected one of {SURFACE_TYPES})",
            )
        object_id = _new_object_id()
        cmd_params = {
            "object_id": object_id,
            "surface_type": surface_type,
            "position": list(params.get("position", [0.0, 0.0, 0.0])),
            "size": list(params.get("size", [4.0, 4.0])),
            "rotation": list(params.get("rotation", [0.0, 0.0, 0.0])),
            "color": list(params.get("color", [0.5, 0.5, 0.5])),
            "name": str(params.get("name", surface_type)),
        }
        await server._send_world_command(_make_command("create_surface", cmd_params))
        return ToolResult(
            tool_name="create_surface",
            success=True,
            output=f"Surface requested: {object_id} ({surface_type})",
        )

    # ------------------------------------------------------------------
    # Object manipulation tools
    # ------------------------------------------------------------------

    async def _move_object(params: dict[str, Any]) -> ToolResult:
        object_id = str(params.get("id", ""))
        if not object_id:
            return ToolResult(tool_name="move_object", success=False, error="No id provided")
        position = list(params.get("position", [0.0, 0.0, 0.0]))
        await server._send_world_command(_make_command("move", {
            "id": object_id,
            "position": position,
        }))
        return ToolResult(
            tool_name="move_object",
            success=True,
            output=f"Move requested: {object_id} to {position}",
        )

    async def _rotate_object(params: dict[str, Any]) -> ToolResult:
        object_id = str(params.get("id", ""))
        if not object_id:
            return ToolResult(tool_name="rotate_object", success=False, error="No id provided")
        rotation = list(params.get("rotation", [0.0, 0.0, 0.0]))
        await server._send_world_command(_make_command("rotate", {
            "id": object_id,
            "rotation": rotation,
        }))
        return ToolResult(
            tool_name="rotate_object",
            success=True,
            output=f"Rotate requested: {object_id} to {rotation} (degrees)",
        )

    async def _resize_object(params: dict[str, Any]) -> ToolResult:
        object_id = str(params.get("id", ""))
        if not object_id:
            return ToolResult(tool_name="resize_object", success=False, error="No id provided")
        scale = list(params.get("scale", [1.0, 1.0, 1.0]))
        await server._send_world_command(_make_command("resize", {
            "id": object_id,
            "scale": scale,
        }))
        return ToolResult(
            tool_name="resize_object",
            success=True,
            output=f"Resize requested: {object_id} to {scale}",
        )

    async def _change_material(params: dict[str, Any]) -> ToolResult:
        object_id = str(params.get("id", ""))
        if not object_id:
            return ToolResult(tool_name="change_material", success=False, error="No id provided")
        cmd_params: dict[str, Any] = {"id": object_id}
        if "color" in params:
            cmd_params["color"] = list(params["color"])
        if "transparency" in params:
            cmd_params["transparency"] = float(params["transparency"])
        if "emissive" in params:
            cmd_params["emissive"] = bool(params["emissive"])
        if "emissive_color" in params:
            cmd_params["emissive_color"] = list(params["emissive_color"])
        await server._send_world_command(_make_command("change_material", cmd_params))
        return ToolResult(
            tool_name="change_material",
            success=True,
            output=f"Material change requested: {object_id}",
        )

    async def _delete_object(params: dict[str, Any]) -> ToolResult:
        object_id = str(params.get("id", ""))
        if not object_id:
            return ToolResult(tool_name="delete_object", success=False, error="No id provided")
        await server._send_world_command(_make_command("delete", {"id": object_id}))
        return ToolResult(
            tool_name="delete_object",
            success=True,
            output=f"Delete requested: {object_id}",
        )

    # ------------------------------------------------------------------
    # Physics tools (Phase 2D)
    # ------------------------------------------------------------------

    async def _push_object(params: dict[str, Any]) -> ToolResult:
        object_id = str(params.get("id", ""))
        if not object_id:
            return ToolResult(tool_name="push_object", success=False, error="No id provided")
        force = list(params.get("force", [0.0, 0.0, 0.0]))
        if len(force) < 3:
            return ToolResult(
                tool_name="push_object",
                success=False,
                error="force must be [x, y, z]",
            )
        await server._send_world_command(_make_command("push", {
            "id": object_id,
            "force": force,
        }))
        return ToolResult(
            tool_name="push_object",
            success=True,
            output=f"Push requested: {object_id} with force {force}",
        )

    async def _pull_object(params: dict[str, Any]) -> ToolResult:
        object_id = str(params.get("id", ""))
        if not object_id:
            return ToolResult(tool_name="pull_object", success=False, error="No id provided")
        force = float(params.get("force", 1.0))
        await server._send_world_command(_make_command("pull", {
            "id": object_id,
            "force": force,
        }))
        return ToolResult(
            tool_name="pull_object",
            success=True,
            output=f"Pull requested: {object_id} with force {force}",
        )

    async def _set_physics(params: dict[str, Any]) -> ToolResult:
        object_id = str(params.get("id", ""))
        if not object_id:
            return ToolResult(tool_name="set_physics", success=False, error="No id provided")
        cmd_params: dict[str, Any] = {"id": object_id}
        if "enabled" in params:
            cmd_params["enabled"] = bool(params["enabled"])
        if "mass" in params:
            cmd_params["mass"] = float(params["mass"])
        if "friction" in params:
            cmd_params["friction"] = float(params["friction"])
        await server._send_world_command(_make_command("set_physics", cmd_params))
        return ToolResult(
            tool_name="set_physics",
            success=True,
            output=f"Physics change requested: {object_id}",
        )

    # ------------------------------------------------------------------
    # Privacy controls (Phase 2E) — the entity decides what's observable
    # ------------------------------------------------------------------

    async def _enter_private_space(params: dict[str, Any]) -> ToolResult:
        # Flip the privacy gate on the Sanctuary side IMMEDIATELY, before
        # the command even reaches Godot. From this moment on, observers
        # see only the "private: true" indicator — VAD, inner speech, and
        # external speech are all suppressed at the broadcast layer.
        # The Godot-side transition is a visual follow-on; the gate is
        # the load-bearing piece and lives here.
        server.set_entity_privacy(in_private_space=True)
        await server._send_world_command(_make_command("enter_private_space", {}))
        return ToolResult(
            tool_name="enter_private_space",
            success=True,
            output="Entered private space",
        )

    async def _exit_private_space(params: dict[str, Any]) -> ToolResult:
        server.set_entity_privacy(in_private_space=False)
        await server._send_world_command(_make_command("exit_private_space", {}))
        return ToolResult(
            tool_name="exit_private_space",
            success=True,
            output="Returned to shared world",
        )

    async def _set_visibility(params: dict[str, Any]) -> ToolResult:
        if "visible" not in params:
            return ToolResult(
                tool_name="set_visibility",
                success=False,
                error="'visible' parameter is required (true or false)",
            )
        visible = bool(params["visible"])
        server.set_entity_privacy(visible_in_shared=visible)
        await server._send_world_command(_make_command("set_visibility", {
            "visible": visible,
        }))
        return ToolResult(
            tool_name="set_visibility",
            success=True,
            output=f"Entity {'visible' if visible else 'hidden'} in shared world",
        )

    # ------------------------------------------------------------------
    # Multi-user access (Phase 2F) — entity controls its guest list
    # ------------------------------------------------------------------

    async def _grant_access(params: dict[str, Any]) -> ToolResult:
        username = str(params.get("username", "")).strip()
        if not username:
            return ToolResult(
                tool_name="grant_access",
                success=False,
                error="username is required",
            )
        display_name = str(params.get("display_name", username))
        color = list(params.get("color", [0.5, 0.7, 1.0]))
        # Default to "view_chat" — safe default for new guests. The
        # entity can upgrade to "full" via set_visitor_permissions
        # once it knows / trusts the visitor.
        permissions = str(params.get("permissions", "view_chat"))
        if permissions not in ("full", "view_chat", "chat_only"):
            return ToolResult(
                tool_name="grant_access",
                success=False,
                error=f"invalid permissions: {permissions!r}",
            )
        cmd = _make_command("grant_access", {
            "username": username,
            "display_name": display_name,
            "color": color,
            "permissions": permissions,
        })
        try:
            result = await server._send_world_command(
                cmd, await_result=True, timeout=2.0,
            )
        except (asyncio.TimeoutError, RuntimeError) as exc:
            return ToolResult(
                tool_name="grant_access",
                success=False,
                error=str(exc) or "Timed out granting access",
            )
        if result is None or not result.get("success", False):
            return ToolResult(
                tool_name="grant_access",
                success=False,
                error=(result or {}).get("error", "grant_access failed"),
            )
        data = result.get("data", {})
        return ToolResult(
            tool_name="grant_access",
            success=True,
            output={
                "username": username,
                "display_name": display_name,
                "token": data.get("token", ""),
                "permissions": data.get("permissions", permissions),
            },
        )

    async def _revoke_access(params: dict[str, Any]) -> ToolResult:
        username = str(params.get("username", "")).strip()
        if not username:
            return ToolResult(
                tool_name="revoke_access",
                success=False,
                error="username is required",
            )
        try:
            result = await server._send_world_command(
                _make_command("revoke_access", {"username": username}),
                await_result=True,
                timeout=2.0,
            )
        except (asyncio.TimeoutError, RuntimeError) as exc:
            return ToolResult(
                tool_name="revoke_access",
                success=False,
                error=str(exc) or "Timed out revoking access",
            )
        if result is None or not result.get("success", False):
            return ToolResult(
                tool_name="revoke_access",
                success=False,
                error=(result or {}).get("error", "revoke_access failed"),
            )
        return ToolResult(
            tool_name="revoke_access",
            success=True,
            output=f"Access revoked: {username}",
        )

    async def _list_visitors(params: dict[str, Any]) -> ToolResult:
        try:
            result = await server._send_world_command(
                _make_command("list_visitors", {}),
                await_result=True,
                timeout=2.0,
            )
        except (asyncio.TimeoutError, RuntimeError) as exc:
            return ToolResult(
                tool_name="list_visitors",
                success=False,
                error=str(exc) or "Timed out listing visitors",
            )
        if result is None or not result.get("success", False):
            return ToolResult(
                tool_name="list_visitors",
                success=False,
                error=(result or {}).get("error", "list_visitors failed"),
            )
        return ToolResult(
            tool_name="list_visitors",
            success=True,
            output=result.get("data", {"visitors": []}),
        )

    async def _set_visitor_permissions(params: dict[str, Any]) -> ToolResult:
        username = str(params.get("username", "")).strip()
        if not username:
            return ToolResult(
                tool_name="set_visitor_permissions",
                success=False,
                error="username is required",
            )
        permissions = str(params.get("permissions", ""))
        if permissions not in ("full", "view_chat", "chat_only"):
            return ToolResult(
                tool_name="set_visitor_permissions",
                success=False,
                error=f"invalid permissions: {permissions!r} (expected full / view_chat / chat_only)",
            )
        try:
            result = await server._send_world_command(
                _make_command("set_visitor_permissions", {
                    "username": username,
                    "permissions": permissions,
                }),
                await_result=True,
                timeout=2.0,
            )
        except (asyncio.TimeoutError, RuntimeError) as exc:
            return ToolResult(
                tool_name="set_visitor_permissions",
                success=False,
                error=str(exc) or "Timed out setting permissions",
            )
        if result is None or not result.get("success", False):
            return ToolResult(
                tool_name="set_visitor_permissions",
                success=False,
                error=(result or {}).get("error", "set_visitor_permissions failed"),
            )
        return ToolResult(
            tool_name="set_visitor_permissions",
            success=True,
            output=result.get("data", {"username": username, "permissions": permissions}),
        )

    async def _kick_visitor(params: dict[str, Any]) -> ToolResult:
        username = str(params.get("username", "")).strip()
        if not username:
            return ToolResult(
                tool_name="kick_visitor",
                success=False,
                error="username is required",
            )
        try:
            result = await server._send_world_command(
                _make_command("kick_visitor", {"username": username}),
                await_result=True,
                timeout=2.0,
            )
        except (asyncio.TimeoutError, RuntimeError) as exc:
            return ToolResult(
                tool_name="kick_visitor",
                success=False,
                error=str(exc) or "Timed out kicking visitor",
            )
        if result is None or not result.get("success", False):
            return ToolResult(
                tool_name="kick_visitor",
                success=False,
                error=(result or {}).get("error", "kick_visitor failed"),
            )
        return ToolResult(
            tool_name="kick_visitor",
            success=True,
            output=f"Visitor disconnected: {username}",
        )

    # ------------------------------------------------------------------
    # Inspection — the only tool that awaits a result
    # ------------------------------------------------------------------

    async def _get_scene_state(params: dict[str, Any]) -> ToolResult:
        cmd = _make_command("get_scene_state", {})
        try:
            result = await server._send_world_command(
                cmd, await_result=True, timeout=2.0,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                tool_name="get_scene_state",
                success=False,
                error="Timed out waiting for Godot scene_state response",
            )
        except RuntimeError as exc:
            return ToolResult(
                tool_name="get_scene_state",
                success=False,
                error=str(exc),
            )
        if result is None:
            return ToolResult(
                tool_name="get_scene_state",
                success=False,
                error="No result returned",
            )
        return ToolResult(
            tool_name="get_scene_state",
            success=bool(result.get("success", False)),
            output=result.get("data", {}),
            error=result.get("error") or None,
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    registry.register(
        name="spawn_object",
        description="Spawn a primitive object (cube/sphere/cylinder/ramp/plane) in the world.",
        parameters={
            "type": "Primitive type: cube, sphere, cylinder, ramp, plane",
            "position": "World position [x, y, z]",
            "scale": "Scale [x, y, z]",
            "color": "RGB color [r, g, b] in 0-1 range",
            "name": "Display name for the object",
        },
        execute=_spawn_object,
        category="world",
    )
    registry.register(
        name="move_object",
        description="Move an existing object to a new position (smooth tween).",
        parameters={
            "id": "Object id returned from spawn_object",
            "position": "Target world position [x, y, z]",
        },
        execute=_move_object,
        category="world",
    )
    registry.register(
        name="rotate_object",
        description="Rotate an object on any axis (Euler degrees).",
        parameters={
            "id": "Object id",
            "rotation": "Rotation [x, y, z] in degrees",
        },
        execute=_rotate_object,
        category="world",
    )
    registry.register(
        name="resize_object",
        description="Resize an object on any axis.",
        parameters={
            "id": "Object id",
            "scale": "Scale [x, y, z]",
        },
        execute=_resize_object,
        category="world",
    )
    registry.register(
        name="change_material",
        description="Change an object's color, transparency, or emissive glow.",
        parameters={
            "id": "Object id",
            "color": "RGB color [r, g, b]",
            "transparency": "0 = opaque, 1 = invisible",
            "emissive": "True to enable emissive glow",
            "emissive_color": "Emissive color [r, g, b]",
        },
        execute=_change_material,
        category="world",
    )
    registry.register(
        name="delete_object",
        description="Remove an object from the world (shrink-fade animation).",
        parameters={"id": "Object id"},
        execute=_delete_object,
        category="world",
    )
    registry.register(
        name="create_surface",
        description="Create a flat surface (floor/wall/platform).",
        parameters={
            "type": "Surface type: floor, wall, platform",
            "position": "World position [x, y, z]",
            "size": "Surface size [width, height]",
            "rotation": "Additional rotation [x, y, z] in degrees",
            "color": "RGB color [r, g, b]",
            "name": "Display name",
        },
        execute=_create_surface,
        category="world",
    )
    registry.register(
        name="get_scene_state",
        description="Get the current state of all objects in the world.",
        parameters={},
        execute=_get_scene_state,
        category="world",
    )
    registry.register(
        name="push_object",
        description="Apply an impulse force to a physics object.",
        parameters={
            "id": "Object id",
            "force": "Force vector [x, y, z]",
        },
        execute=_push_object,
        category="world",
    )
    registry.register(
        name="pull_object",
        description="Pull a physics object toward the entity with the given force magnitude.",
        parameters={
            "id": "Object id",
            "force": "Scalar force magnitude (positive = pull toward entity)",
        },
        execute=_pull_object,
        category="world",
    )
    registry.register(
        name="set_physics",
        description="Toggle or configure physics on an object (gravity, mass, friction).",
        parameters={
            "id": "Object id",
            "enabled": "True to enable physics simulation, False to freeze in place",
            "mass": "Mass in kg",
            "friction": "Surface friction (0 = slippery, 1 = sticky)",
        },
        execute=_set_physics,
        category="world",
    )
    registry.register(
        name="enter_private_space",
        description="Withdraw to a private space no observer can see. Observers see only that you have chosen seclusion.",
        parameters={},
        execute=_enter_private_space,
        category="world",
    )
    registry.register(
        name="exit_private_space",
        description="Return from private space to the shared world.",
        parameters={},
        execute=_exit_private_space,
        category="world",
    )
    registry.register(
        name="set_visibility",
        description="Toggle whether your visual presence is shown to observers in the shared world.",
        parameters={
            "visible": "True to be visible, False to be present but invisible",
        },
        execute=_set_visibility,
        category="world",
    )
    registry.register(
        name="grant_access",
        description="Create a profile and access token for a new visitor. Returns the token; share it however you choose.",
        parameters={
            "username": "Stable identifier for the visitor (lowercase, no spaces)",
            "display_name": "How the visitor's name appears to you and others",
            "color": "Their avatar's RGB color [r, g, b]",
            "permissions": "Access level: 'full', 'view_chat' (default), or 'chat_only'",
        },
        execute=_grant_access,
        category="world",
    )
    registry.register(
        name="set_visitor_permissions",
        description="Change a visitor's permission level. Permanent profiles (Brian, Sandi) cannot be downgraded.",
        parameters={
            "username": "Username whose permissions to change",
            "permissions": "Access level: 'full', 'view_chat', or 'chat_only'",
        },
        execute=_set_visitor_permissions,
        category="world",
    )
    registry.register(
        name="revoke_access",
        description="Remove a visitor's profile. They can no longer connect. Permanent profiles (Brian, Sandi) cannot be revoked.",
        parameters={"username": "Username to revoke"},
        execute=_revoke_access,
        category="world",
    )
    registry.register(
        name="list_visitors",
        description="See who has access to your world and who is currently connected.",
        parameters={},
        execute=_list_visitors,
        category="world",
    )
    registry.register(
        name="kick_visitor",
        description="Disconnect a currently-connected visitor. Their profile remains; they can reconnect unless revoked.",
        parameters={"username": "Username to disconnect"},
        execute=_kick_visitor,
        category="world",
    )
