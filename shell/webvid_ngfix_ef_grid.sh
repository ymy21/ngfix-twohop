#!/bin/bash
# Shared defaults, copied from upstream NGFix TestQueries' K-specific lists.
# A global EF_VALUES or K-specific EF_VALUES_K10/K100 can still override them.
webvid_select_ef_grid() {
    if [[ -n ${EF_VALUES:-} ]]; then
        printf '%s\n' "${EF_VALUES}"
        return
    fi
    case "$1" in
        10) printf '%s\n' "${EF_VALUES_K10:-10,15,20,30,40,50,60,70,80,90,100,120,150,180,200,300,400,500,800,1000}" ;;
        100) printf '%s\n' "${EF_VALUES_K100:-100,110,120,130,140,150,160,180,200,250,300,400,500,800,1000,2000}" ;;
        *) echo "K must be 10 or 100" >&2;return 1 ;;
    esac
}
