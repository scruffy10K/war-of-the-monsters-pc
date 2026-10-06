using System;
using System.IO;
using System.Security.Cryptography;

// Read receipts from older local installations; player setup never builds code.
static class GameBuild
{
    public static string InstalledExecutable(string root)
    {
        string active = Path.Combine(root, "game", "active.txt");
        if (!File.Exists(active)) return null;
        string[] lines = File.ReadAllLines(active);
        if (lines.Length != 3 || lines[0] != "WOTM-GAME-1" || !Hex(lines[1],24) || !Hex(lines[2],64))
            throw new IOException("The local game installation record is invalid. Extract the complete Windows download into a new folder.");
        string exe = Path.Combine(root, "game", "versions", lines[1], "War of the Monsters.exe");
        if (!File.Exists(exe)) return null;
        using (var hash = SHA256.Create()) using (var stream = File.OpenRead(exe)) {
            string actual = BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
            if (actual != lines[2]) throw new IOException("The local game executable has changed. Restore its matching build or extract the complete Windows download into a new folder.");
        }
        return exe;
    }

    static bool Hex(string value, int count)
    {
        if (value.Length != count) return false;
        foreach (char ch in value) if (!(ch >= '0' && ch <= '9') && !(ch >= 'a' && ch <= 'f')) return false;
        return true;
    }

}
