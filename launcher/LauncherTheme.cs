using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Windows.Forms;

static class MovieTheme
{
    public static readonly Color Ink = Color.FromArgb(18, 18, 16);
    public static readonly Color Panel = Color.FromArgb(53, 53, 47);
    public static readonly Color Field = Color.FromArgb(32, 33, 30);
    public static readonly Color Line = Color.FromArgb(104, 101, 87);
    public static readonly Color Paper = Color.FromArgb(246, 236, 211);
    public static readonly Color Muted = Color.FromArgb(190, 185, 166);
    public static readonly Color Gold = Color.FromArgb(217, 173, 105);
    public static readonly Color Red = Color.FromArgb(180, 54, 43);

    [DllImport("dwmapi.dll")] static extern int DwmSetWindowAttribute(IntPtr handle, int attribute, ref int value, int size);
    public static void DarkTitle(Form form)
    {
        try { int enabled = 1; DwmSetWindowAttribute(form.Handle, 20, ref enabled, sizeof(int)); }
        catch (DllNotFoundException) { } catch (EntryPointNotFoundException) { }
    }
    public static void PaintScene(Graphics g, MenuArtwork art)
    {
        g.SmoothingMode = SmoothingMode.AntiAlias;
        g.InterpolationMode = InterpolationMode.HighQualityBicubic;
        if (art != null) g.DrawImage(art.Sky, new Rectangle(0, 0, 1120, 780));
        else using (var sky = new LinearGradientBrush(new Rectangle(0,0,1120,780),
            Color.FromArgb(48,49,54), Color.FromArgb(104,62,48), LinearGradientMode.Vertical))
            g.FillRectangle(sky, 0,0,1120,780);
        using (var dusk = new SolidBrush(Color.FromArgb(70, 10, 11, 12))) g.FillRectangle(dusk, 0,0,1120,780);
        if (art != null) {
            g.DrawImage(art.Skyline, new Rectangle(0, 639, 480, 141));
            g.DrawImage(art.Skyline, new Rectangle(640, 639, 480, 141));
        }
        using (var earth = new LinearGradientBrush(new Rectangle(0,735,1120,45),
            Color.FromArgb(135,19,21,17), Ink, LinearGradientMode.Vertical))
            g.FillRectangle(earth, 0,735,1120,45);
        // The original menu is a drive-in projection screen on a steel frame.
        using (var shadow = new Pen(Color.FromArgb(16,17,15), 13)) {
            g.DrawLine(shadow, 94,28,84,780); g.DrawLine(shadow, 1026,28,1036,780);
            g.DrawLine(shadow, 90,626,34,780); g.DrawLine(shadow, 1030,626,1086,780);
            g.DrawLine(shadow, 78,752,1042,752);
        }
        using (var steel = new Pen(Color.FromArgb(57,57,49), 3)) {
            g.DrawLine(steel, 90,42,80,774); g.DrawLine(steel, 1030,42,1040,774);
            g.DrawLine(steel, 89,640,38,777); g.DrawLine(steel, 1031,640,1082,777);
        }
        Rectangle screen = new Rectangle(67,25,986,716);
        // Keep the player's imported sky visible inside the projector frame.
        // Both the backing and the cloth used to be opaque, hiding the scene.
        using (var outer = new Pen(Ink, 5)) g.DrawRectangle(outer, 64,22,992,722);
        if (art != null) using (var cloth = new ImageAttributes()) {
            ColorMatrix opacity = new ColorMatrix(); opacity.Matrix33 = 0.08f;
            cloth.SetColorMatrix(opacity);
            g.DrawImage(art.Screen, screen, 0,0,art.Screen.Width,art.Screen.Height,GraphicsUnit.Pixel,cloth);
        }
        using (var wash = new SolidBrush(Color.FromArgb(86, 12, 13, 14))) g.FillRectangle(wash, screen);
        using (var light = new LinearGradientBrush(screen, Color.FromArgb(15,232,224,195),
            Color.FromArgb(0,232,224,195), LinearGradientMode.Vertical)) g.FillRectangle(light, screen);
        using (var edge = new Pen(Color.FromArgb(163,160,140), 2)) g.DrawRectangle(edge, 64,22,992,722);
        using (var inner = new Pen(Color.FromArgb(72,72,63))) g.DrawRectangle(inner, 69,27,982,712);
        if (art != null) g.DrawImage(art.Logo, new Rectangle(390,32,340,170));
        else {
            using (var font = new Font("Impact", 34))
            using (var text = new SolidBrush(Gold))
            using (var format = new StringFormat { Alignment=StringAlignment.Center,LineAlignment=StringAlignment.Center })
                g.DrawString("WAR OF THE\nMONSTERS",font,text,new RectangleF(340,47,440,142),format);
        }
        using (var font = new Font("Segoe UI",9,FontStyle.Bold))
        using (var text = new SolidBrush(Muted))
        using (var format = new StringFormat { Alignment=StringAlignment.Center,LineAlignment=StringAlignment.Center })
            g.DrawString("PC EDITION  /  GAME SETTINGS",font,text,new RectangleF(300,198,520,21),format);
        using (var line = new Pen(Color.FromArgb(102,100,87))) {
            g.DrawLine(line,112,218,420,218); g.DrawLine(line,700,218,1008,218);
            g.DrawLine(line,112,555,1008,555); g.DrawLine(line,112,665,1008,665);
        }
        if (art != null) {
            for (int x=0; x<1120; x+=110) g.DrawImage(art.Grass,new Rectangle(x,748,110,32));
        }
    }
}

sealed class MovieHelp : Label
{
    public MovieHelp() { AutoSize=false; Cursor=Cursors.Help; TabStop=false; BackColor=Color.Transparent; }
    protected override void OnPaint(PaintEventArgs e)
    {
        e.Graphics.SmoothingMode=SmoothingMode.AntiAlias;
        using(var pen=new Pen(MovieTheme.Line)) e.Graphics.DrawEllipse(pen,1,1,Width-3,Height-3);
        TextRenderer.DrawText(e.Graphics,"?",Font,ClientRectangle,MovieTheme.Gold,
            TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter | TextFormatFlags.NoPadding);
    }
}

sealed class MovieButton : Button
{
    public bool Primary;
    bool hovered,pressed;
    public MovieButton()
    {
        FlatStyle=FlatStyle.Flat; FlatAppearance.BorderSize=0;
        SetStyle(ControlStyles.UserPaint | ControlStyles.AllPaintingInWmPaint | ControlStyles.OptimizedDoubleBuffer,true);
        Cursor=Cursors.Hand;
    }
    protected override void OnMouseEnter(EventArgs e) { hovered=true;Invalidate();base.OnMouseEnter(e); }
    protected override void OnMouseLeave(EventArgs e) { hovered=false;pressed=false;Invalidate();base.OnMouseLeave(e); }
    protected override void OnMouseDown(MouseEventArgs e) { pressed=e.Button==MouseButtons.Left;Invalidate();base.OnMouseDown(e); }
    protected override void OnMouseUp(MouseEventArgs e) { pressed=false;Invalidate();base.OnMouseUp(e); }
    protected override void OnGotFocus(EventArgs e) { Invalidate();base.OnGotFocus(e); }
    protected override void OnLostFocus(EventArgs e) { Invalidate();base.OnLostFocus(e); }
    protected override void OnKeyDown(KeyEventArgs e) { if(e.KeyCode==Keys.Space){pressed=true;Invalidate();}base.OnKeyDown(e); }
    protected override void OnKeyUp(KeyEventArgs e) { pressed=false;Invalidate();base.OnKeyUp(e); }
    protected override void OnPaint(PaintEventArgs e)
    {
        Graphics g=e.Graphics;
        Color fill=!Enabled?MovieTheme.Panel:Primary?
            (pressed?Color.FromArgb(134,41,32):hovered?Color.FromArgb(205,70,52):MovieTheme.Red):
            (pressed?MovieTheme.Panel:hovered?Color.FromArgb(70,68,56):MovieTheme.Field);
        using(var brush=new SolidBrush(fill)) g.FillRectangle(brush,ClientRectangle);
        using(var border=new Pen(Focused?MovieTheme.Gold:Primary?Color.FromArgb(213,114,68):MovieTheme.Line,Focused?2:1))
            g.DrawRectangle(border,1,1,Width-3,Height-3);
        Rectangle text=new Rectangle(8,0,Width-(Primary?40:16),Height);
        TextRenderer.DrawText(g,Text,Font,text,Enabled?MovieTheme.Paper:MovieTheme.Muted,
            TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter | TextFormatFlags.SingleLine | TextFormatFlags.EndEllipsis);
        if(Primary) using(var brush=new SolidBrush(Enabled?MovieTheme.Paper:MovieTheme.Muted))
            g.FillPolygon(brush,new[]{new Point(Width-29,Height/2-6),new Point(Width-21,Height/2),new Point(Width-29,Height/2+6)});
    }
}
