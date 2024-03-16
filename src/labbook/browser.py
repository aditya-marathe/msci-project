"""\
src / labbook / browser.py
--------------------------------------------------------------------------------

Aditya Marathe
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import unicode_literals
from __future__ import print_function

__all__ = [
    'LabBookApplication'
]

from typing import Any

import pathlib

import json

from datetime import datetime

import tkinter as tk
from tkinter import ttk

from labbook.logging import _process_dir


TITLE_FONT = ('Freestyle Script', 25)
SUBTITLE_FONT = ('Segoe UI Semibold', 15)
TEXT_FONT = ('Segoe UI Light', 10)
BUTTON_FONT = ('Segoe UI Light', 10, 'bold')


# Browser widgets --------------------------------------------------------------

class _AutoHideScrollbar(ttk.Scrollbar):
    """\
    _AutoHideScrollbar
    ------------------

    Private class.

    A scrollbar that automatically hides when the content fits on the frame.
    """
    def set(self, first: float, last: float):
        """\
        Overrides the behaviour of the `set` method for this `ttk.Scrollbar`
        instance.
        """
        if float(first) <= 0. and float(last) >= 1.:
            self.pack_forget()
        else:
            self.pack(fill=tk.Y, side=tk.RIGHT)

        ttk.Scrollbar.set(self, first, last)


class _ScrollableFrame(tk.Frame):
    """\
    _ScrollableFrame
    ----------------

    Private class.

    Wraps a `tk.Frame` to create a frame with a scrollbar.
    """
    def __init__(self, *args, **kwargs):
        """\
        Initialises a `_ScrollableFrame`.

        Args
        ----
        *frame_args, **frame_kwargs
            Arguments for `tk.Frame`.
        """
        super().__init__(*args, **kwargs)

        # Canvas (required for scrollbar widget...)
        self._canvas = tk.Canvas(
            self,
            bg=kwargs.get('bg', None) or kwargs.get('background', None),
            bd=0,
            highlightthickness=0
        )
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Container
        self.container = tk.Frame(self._canvas, bd=0, highlightthickness=0)
        _canvas_window = self._canvas.create_window(
            (0, 0), 
            window=self.container, 
            anchor=tk.NW
        )

        # Scrollbar
        self._scrollbar = _AutoHideScrollbar(
            self, 
            orient=tk.VERTICAL, 
            command=self._canvas.yview
        )

        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        self.container.bind(
            '<Configure>', 
            lambda event: self._canvas.configure(
                scrollregion=self._canvas.bbox(tk.ALL)
            )
        )
        self._canvas.bind(
            '<Configure>',
            lambda event: self._canvas.itemconfig(
                _canvas_window,
                width=event.width
            )
        )


class _AutoWrapLabel(tk.Label):
    """\
    _AutoWrapLabel
    --------------

    Private class.
    
    A label which automatically wraps to fit text inside of a frame.
    """
    def __init__(self, *args, **kwargs):
        """\
        Initialises a `_AutoWrapLabel`.

        Args
        ----
        *label_args, **label_kwargs
            Arguments for `tk.Label`.

        """
        super().__init__(*args, **kwargs)

        # Auto-wrap behaviour
        self.bind(
            '<Configure>', 
            lambda event: self.configure(wraplength=self.winfo_width())
        )


class _CustomButton(tk.Button):
    """\
    _CustomButton
    -------------

    Private class.

    Button which changes colour when hovered.    
    """
    def __init__(
            self,
            *args,
            hover_bg: str | None = None,
            hover_fg: str | None = None,
            **kwargs
        ) -> None:
        """\
        Initialises a `_CustomButton`.

        Args
        ----
        hover_bg: str | None
            Button background colour when hovered.
        hover_fg: str | None
            Button foreground (text) colour when hovered.
        *button_args, **button_kwargs
            Arguments for `tk.Button`.
        """
        super().__init__(*args, **kwargs)

        self._default_bg, self._default_fg = self["bg"], self["fg"]

        self.hover_bg = hover_bg if hover_bg is not None else self["bg"]
        self.hover_fg = hover_fg if hover_fg is not None else self["fg"]

        self.bind("<Enter>", self._on_hover_event)
        self.bind("<Leave>", self._on_leave_event)

        self._command = kwargs.get("command", lambda: 0)
        self.bind("<Button-1>", self._exec_command, add=True)

    def _on_hover_event(self, *_) -> None:
        """\
        Changes button colour when triggered by a hover event.
        """
        self["bg"] = self.hover_bg
        self["fg"] = self.hover_fg

    def _on_leave_event(self, *_) -> None:
        """\
        Changes button colour back to normal when triggered by a leave event.
        """
        self["bg"] = self._default_bg
        self["fg"] = self._default_fg

    def _exec_command(self, *_) -> str:
        """\
        Wraps the command provided by the user to modify behaviour of 
        `tk.Button` by using some cheeky backend magic.
        """
        if self['state'] == tk.NORMAL:
            self._command()
        
        self._on_hover_event()

        return "break"


class _LabBookView(tk.Toplevel):
    """
    _LabBookView
    ------------

    Private class.

    Wraps behaviour of the `tk.Toplevel` to open a window (with set formatting
    / layout) which displays information about the model.
    """
    def __init__(self, *args, config_dict: dict[str, Any], **kwargs) -> None:
        """
        Initialises `_LabBookView`.

        Args
        ----
        config_dict: dict[str, Any]
            The specified configuration for the model.
        *toplevel_args, **toplevel_kwargs
            Arguments for `tk.Toplevel`.
        """
        super().__init__(*args, **kwargs)

        self.config_dict = config_dict

        SCREEN_WIDTH = self.winfo_screenwidth()
        SCREEN_HEIGHT = self.winfo_screenheight()
        WINDOW_SIZE = 820, 600

        self.title('LabBook | Model View')
        self.geometry(
            '{}x{}+{}+{}'.format(
                WINDOW_SIZE[0],
                WINDOW_SIZE[1],
                int(0.94 * (SCREEN_WIDTH - WINDOW_SIZE[0])),
                int(0.22 * (SCREEN_HEIGHT - WINDOW_SIZE[1]))
            )
        )

        self.protocol('WM_DELETE_WINDOW', self._on_destroy)

        self.configure(bg='white')

        self.grid_columnconfigure(0, weight=1)
        # self.grid_columnconfigure(1, weight=1)

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        # self.grid_rowconfigure(2, weight=1)

        # Content
        top_container = tk.Frame(
            self,
            bg='light gray'
        )
        top_container.grid(
            row=0, 
            column=0,
            columnspan=2, 
            sticky=tk.NSEW
        )

        _AutoWrapLabel(
            top_container, 
            text='Model View',
            justify=tk.LEFT,
            anchor=tk.W,
            font=TITLE_FONT,
            bg='light gray'
        ).pack(
            padx=10,
            pady=10,
            side=tk.LEFT,
            anchor=tk.W
        )

        _AutoWrapLabel(
            top_container, 
            text=self.config_dict['Time'],
            justify=tk.RIGHT,
            anchor=tk.E,
            font=SUBTITLE_FONT,
            bg='light gray'
        ).pack(
            padx=10,
            pady=10,
            side=tk.RIGHT,
            anchor=tk.E
        )

        scrollable_frame = _ScrollableFrame(self, bg='white')
        container = scrollable_frame.container

        container.configure(bg='white')
        
        scrollable_frame.grid(
            row=1,
            column=0,
            sticky=tk.NSEW
        )

        comments_section = tk.LabelFrame(
            container,
            text='Comments',
            font=BUTTON_FONT,
            bg='white'
        )
        comments_section.grid(
            row=1, 
            column=0, 
            sticky=tk.NS,
            padx=10,
            pady=10
        )
        
        comments_section.grid_rowconfigure(0, weight=1)
        comments_section.grid_rowconfigure(1, weight=0)
        comments_section.grid_columnconfigure(0, weight=1)

        self.comments_text = tk.Text(
            comments_section,
            relief=tk.FLAT,
            font=TEXT_FONT,
            bg='white'
        )
        self.comments_text.grid(
            row=0,
            column=0,
            sticky=tk.NS,
            padx=10,
            pady=10
        )

        self.comments_text.insert(tk.END, self.config_dict['Comments'])
        self.comments_text['state'] = tk.DISABLED

        #
        self.main_comment_container = tk.Frame(comments_section, bg='white')
        self.main_comment_container.grid(row=1, column=0, sticky=tk.NSEW)

        _CustomButton(
            self.main_comment_container,
            text='Edit',
            relief=tk.FLAT,
            font=BUTTON_FONT,
            bg='white',
            hover_bg='light gray',
            command=self._edit_comment
        ).pack(
            pady=(0, 10)
        )
        
        #
        self.editor_container = tk.Frame(comments_section, bg='white')

        _CustomButton(
            self.editor_container,
            text='Save',
            relief=tk.FLAT,
            font=BUTTON_FONT,
            bg='white',
            hover_bg='light gray',
            command=self._save_edited_comment
        ).pack(
            pady=(0, 10)
        )

        self.focus_force()

    def _on_destroy(self) -> None:
        """\
        Triggers on a destroy event, forces a refresh on the parent window so 
        that the log comments are updated.
        """
        self.master.load_content()  # type: ignore
        self.destroy()

    def _update_comment(self, updated_comment: str) -> None:
        """\
        Updates the comment on the lab book entry.
        """      
        with open(self.config_dict['LogDir'], 'r+', encoding='utf-8') as file:
            config_dict = json.load(file)
            
            # Update field
            config_dict['Comments'] = updated_comment

            # Update file
            file.seek(0)
            json.dump(config_dict, file, indent=4)
            file.truncate()

        self.config_dict = config_dict

    def _save_edited_comment(self) -> None:
        """
        Saves an edited comment to the log entry.
        """
        self.main_comment_container.grid(row=1, column=0, sticky=tk.NSEW)
        self.editor_container.grid_forget()

        self._update_comment(self.comments_text.get(1.0, tk.END))

        self.comments_text.delete(1.0, tk.END)
        self.comments_text.insert(tk.END, self.config_dict['Comments'])

        self.comments_text['state'] = tk.DISABLED
    
    def _edit_comment(self) -> None:
        """
        Switches to edit mode.
        """
        self.editor_container.grid(row=1, column=0, sticky=tk.NSEW)
        self.main_comment_container.grid_forget()

        self.comments_text['state'] = tk.NORMAL


class _LabBookItem(tk.Frame):
    """\
    _LabBookItem
    ------------

    Lab book log displayed on the browser window.
    """
    def __init__(
            self,
            parent: Any,
            config_dict: dict[str, Any]
        ) -> None:
        """\
        Initialises a `_LabBookItem`.

        Args
        ----
        parent: tk.Frame | tk.Mis
        """
        self.config_dict = config_dict

        super().__init__(
            parent,
            bg='white',
            bd=1,
            highlightthickness=1
        )

        # Content
        self._subtitle_label = _AutoWrapLabel(
            self, 
            text=self.config_dict['Time'],
            font=SUBTITLE_FONT,
            bg='white'
        )
        self._subtitle_label.pack(
            side=tk.TOP,
            fill=tk.BOTH,
            expand=True,
            padx=5
        )

        ttk.Separator(
            self,
            orient=tk.HORIZONTAL
        ).pack(
            fill=tk.X,
            side=tk.TOP,
            padx=5,
            pady=5
        )

        _AutoWrapLabel(
            self,
            text=self.config_dict['Comments'][:100] + ' [...]',
            justify=tk.LEFT,
            anchor=tk.W,
            font=TEXT_FONT,
            bg='white'
        ).pack(
            side=tk.TOP,
            anchor=tk.W,
            fill=tk.BOTH,
            expand=True,
            padx=5
        )

        button_container = tk.Frame(
            self, 
            bg='white'
        )
        button_container.pack(
            side=tk.TOP,
            pady=10
        )

        self._flag_button = _CustomButton(
            button_container,
            text='',
            relief=tk.FLAT,
            font=BUTTON_FONT,
            bg='white',
            hover_bg='light gray',
            command=self._toggle_flag
        )
        self._flag_button.pack(
            side=tk.LEFT,
            padx=(10, 5)
        )

        self.view_button = _CustomButton(
            button_container,
            text='Go to view',
            relief=tk.FLAT,
            font=BUTTON_FONT,
            bg='white',
            hover_bg='light gray',
            command=self._view_command
        )
        self.view_button.pack(side=tk.LEFT, padx=(5, 10))

        self._update_appearance()

    def _view_command(self):
        """\
        On user click event for view button.
        """
        self.view_button['state'] = tk.DISABLED
        _LabBookView(config_dict=self.config_dict)

    def _toggle_flag(self):
        """\
        On user click event for the toggle flag button.
        """
        with open(self.config_dict['LogDir'], 'r+', encoding='utf-8') as file:
            config_dict = json.load(file)
            
            # Update field
            config_dict['Flagged'] = not config_dict['Flagged']

            # Update file
            file.seek(0)
            json.dump(config_dict, file, indent=4)
            file.truncate()

        self.config_dict = config_dict

        self._update_appearance()

    def _update_appearance(self) -> None:
        """\
        Updates the appearance of the `_LabBookItem` if it is flagged.
        """
        is_flagged = self.config_dict['Flagged']

        self._subtitle_label.configure(
            fg='black' if not is_flagged else 'red'
        )
        self._flag_button.configure(
            text='Flag' if not is_flagged else 'Unflag'
        )
        self.configure(
            highlightbackground='gray' if not is_flagged else 'red'
        )


# Browser ----------------------------------------------------------------------


class LabBookApplication(tk.Tk):
    """\
    
    """
    def __init__(
            self,
            lb_dir: str | pathlib.Path
        ) -> None:
        """\
        
        """
        # Validate inputs
        self._lb_dir = _process_dir(lb_dir)

        self._log_dir = _process_dir(self._lb_dir / 'logs')

        # Initialise Tkinter
        super().__init__()

        SCREEN_WIDTH: int = self.winfo_screenwidth()
        SCREEN_HEIGHT: int = self.winfo_screenheight()
        WINDOW_SIZE = 400, 600

        self.title('LabBook | Browser')
        self.geometry(
            '{}x{}+{}+{}'.format(
                WINDOW_SIZE[0],
                WINDOW_SIZE[1],
                int(0.02 * (SCREEN_WIDTH - WINDOW_SIZE[0])),
                int(0.22 * (SCREEN_HEIGHT - WINDOW_SIZE[1]))
            )
        )
        # self.resizable(False, False)

        self.configure(bg='white')

        # Content
        top_container = tk.Frame(
            self,
            bg='light gray'
        )
        top_container.pack(
            fill=tk.X
        )

        _AutoWrapLabel(
            top_container, 
            text='Browser',
            justify=tk.LEFT,
            anchor=tk.W,
            font=TITLE_FONT,
            bg='light gray'
        ).pack(
            padx=10,
            pady=10,
            side=tk.LEFT,
            anchor=tk.W
        )

        _CustomButton(
            top_container,
            text='Refresh',
            relief=tk.FLAT,
            font=BUTTON_FONT,
            bg='light gray',
            hover_bg='gray',
            hover_fg='white',
            command=self.load_content
        ).pack(
            padx=10,
            pady=10,
            side=tk.RIGHT,
            anchor=tk.E
        )

        self.parent_container = _ScrollableFrame(self, bg='white')
        self.parent_container.pack(fill=tk.BOTH, expand=True)

        self.container = self.parent_container.container
        self.container.configure(bg='white')

        self.load_content()

        self.lift()
        self.focus_force()

    def load_content(self) -> None:
        """\

        """
        # Clear the container
        for child in self.container.winfo_children():
            child.destroy()

        # Loop through the logs...
        log_files = sorted(list(self._log_dir.glob('*.json')))

        if not log_files:
            _AutoWrapLabel(
                self.container, 
                text='No records found for display.',
                font=TEXT_FONT,
                bg='white'
            ).pack(
                pady=50,
                fill=tk.BOTH,
                expand=True,
                anchor=tk.CENTER
            )

        for i, log_file in enumerate(log_files):
            with open(log_file, 'r') as file:
                log_data = json.load(file)

            top_padding = 10 if not i else 0

            _LabBookItem(
                parent=self.container,
                config_dict=log_data
            ).pack(
                padx=10, 
                pady=(top_padding, 10),
                fill=tk.X,
                expand=True
            )
