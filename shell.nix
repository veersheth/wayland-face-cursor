{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
    python3Packages.pip
    wtype
    wayland-utils
    
    stdenv.cc
    linuxHeaders
  ];

  shellHook = ''
    if [ ! -d ".venv" ]; then
      python -m venv .venv
    fi
    source .venv/bin/activate
    
    pip install --upgrade pip
    
    # Point pip to the Linux kernel headers provided by Nix
    export C_INCLUDE_PATH="${pkgs.linuxHeaders}/include:$C_INCLUDE_PATH"
    
    pip install wayland-automation
    
    echo "Wayland automation Python environment ready."
  '';
}

