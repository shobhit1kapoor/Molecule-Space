declare module "smiles-drawer" {
  interface DrawerOptions {
    width?: number;
    height?: number;
    bondThickness?: number;
    bondLength?: number;
    padding?: number;
    fontSizeLarge?: number;
    fontSizeSmall?: number;
    compactDrawing?: boolean;
    experimentalSSSR?: boolean;
  }

  interface SvgDrawer {
    draw(tree: unknown, target: string | SVGSVGElement, themeName?: string, infoOnly?: boolean): void;
  }

  const SmilesDrawer: {
    SvgDrawer: new (options?: DrawerOptions) => SvgDrawer;
    parse(smiles: string, successCallback: (tree: unknown) => void, errorCallback?: (error: unknown) => void): void;
  };

  export default SmilesDrawer;
}
