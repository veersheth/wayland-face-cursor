{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
    python3Packages.pip
    wtype
    wayland-utils
    
    stdenv.cc
    linuxHeaders

    # required by mediapipe and opencv-python
    libGL
    glib
    zlib
    stdenv.cc.cc.lib  # provides libstdc++.so.6
    xorg.libxcb
    xorg.xcbutil
    xorg.xcbutilimage
    xorg.xcbutilkeysyms
    xorg.xcbutilrenderutil
    xorg.xcbutilwm
    xorg.libX11
    xorg.libXext
    xorg.libSM
    xorg.libICE
    libxkbcommon
  ];

  shellHook = ''
    if [ ! -d ".venv" ]; then
      python -m venv .venv
    fi
    source .venv/bin/activate

    # Point pip to the Linux kernel headers provided by Nix
    export C_INCLUDE_PATH="${pkgs.linuxHeaders}/include:$C_INCLUDE_PATH"

    # Expose native libs needed by mediapipe and opencv
    export LD_LIBRARY_PATH="${pkgs.libGL}/lib:${pkgs.glib.out}/lib:${pkgs.zlib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.xorg.libxcb}/lib:${pkgs.xorg.xcbutil}/lib:${pkgs.xorg.xcbutilimage}/lib:${pkgs.xorg.xcbutilkeysyms}/lib:${pkgs.xorg.xcbutilrenderutil}/lib:${pkgs.xorg.xcbutilwm}/lib:${pkgs.xorg.libX11}/lib:${pkgs.xorg.libXext}/lib:${pkgs.xorg.libSM}/lib:${pkgs.xorg.libICE}/lib:${pkgs.libxkbcommon}/lib:$LD_LIBRARY_PATH"

    # Only install packages if not already present (avoids network hits on re-entry)
    if ! python -c "import cv2, mediapipe" 2>/dev/null; then
      pip install --upgrade pip
      pip install wayland-automation mediapipe opencv-python
    fi

    echo "Wayland automation Python environment ready."
  '';
}

