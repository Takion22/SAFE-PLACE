from core.safeplace_engine import SafePlaceEngine


def main():

    engine = SafePlaceEngine()

    print("=" * 70)
    print("SAFEPLACE CORE TEST")
    print("=" * 70)

    print("\nSTATUS")
    print(engine.status())

    print("\nFR")
    print(
        engine.route_text(
            "Cette vidéo est intéressante."
        )
    )

    print("\nEN")
    print(
        engine.route_text(
            "This video is interesting."
        )
    )

    print("\nMG")
    print(
        engine.route_text(
            "Mahafinaritra be ity horonantsary ity."
        )
    )

    print("\nRABBIT HOLE")

    metrics = {
        "sequence_length": 120,
        "click_depth": 120,
        "session_time": 5000,
        "unique_videos": 90,
        "repetition_score": 0.30,
        "diversity_score": 0.70
    }

    print(
        engine.analyze_rabbit_hole(
            metrics
        )
    )


if __name__ == "__main__":
    main()