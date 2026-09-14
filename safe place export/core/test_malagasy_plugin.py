from core.malagasy_plugin import MalagasyPlugin


def main():
    plugin = MalagasyPlugin()

    print("=" * 70)
    print("MALAGASY PLUGIN TEST")
    print("=" * 70)

    print("\nCAPABILITIES")
    print(plugin.capabilities())

    tests = [
        ("mg", "Mahafinaritra be ity horonantsary ity."),
        ("fr+mg", "Cette vidéo dia mahafinaritra."),
        ("en+mg", "This video dia tsara."),
    ]

    for language, text in tests:
        print("\nTEST")
        print("language:", language)
        print("supports:", plugin.supports(language))
        print("result:", plugin.process(text, language=language))


if __name__ == "__main__":
    main()
