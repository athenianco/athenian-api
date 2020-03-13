import sys
import traceback

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from athenian.api.models.metadata.github import Base


def main() -> int:
    """Try to fetch one example of each model in the metadata schema from a real DB instance."""
    engine = create_engine(sys.argv[1])
    print("Checking the metadata schema...", flush=True)
    errors = []
    for model in Base._decl_class_registry.values():
        session = sessionmaker(bind=engine)()
        try:
            try:
                model.__name__, model.__tablename__
            except AttributeError:
                continue
            try:
                session.query(model).first()
            except Exception:
                errors.append((model, traceback.format_exc()))
                status = "❌"
            else:
                status = "✔️"
            print("%s  github.%s / %s" % (status, model.__name__, model.__tablename__), flush=True)
        finally:
            session.close()
    for model, exc in errors:
        print("=" * 80, file=sys.stderr)
        print("github.%s / %s\n" % (model.__name__, model.__tablename__), file=sys.stderr)
        print(exc, file=sys.stderr)
        print(file=sys.stderr)
    if not errors:
        # Synchronization level: 100%.
        from random import choice
        print()
        print(choice(nge_phrases))
    return int(bool(errors))


nge_phrases = [
    "Those who hate themselves, cannot love or trust others.",
    "I still don't know where to find happiness. But I'll continue to think about whether it's "
    "good to be here... whether it was good to have been born. But in the end, it's just "
    "realizing the obvious over and over again. Because I am myself.",
    "Mankind's greatest fear is Mankind itself.",
    "If you know pain and hardship, it's easier to be kind to others.",
    "Humans constantly feel pain in their hearts. Because the heart is so sensitive to pain, "
    "humans also feel that to live is to suffer.",
    "Understanding 100% of everything is impossible. That's why we spend all our lives trying to "
    "understand the thinking of others. That's what makes life so interesting.",
    "So fucking what if I'm not you?! That doesn't mean it's okay for you to give up! If you do, "
    "I'll never forgive you as long as I live. God knows I'm not perfect either. I've made tons "
    "of stupid mistakes and later I regretted them. And I've done it over and over again, "
    "thousands of times. A cycle of hollow joy and vicious self-hatred. But even so, every time I "
    "learned something about myself.",
    "Man fears the darkness, and so he scrapes away at the edges of it with fire. He creates life "
    "by diminishing the Darkness.",
    "The interaction of men and women isn't very logical.",
    "This is your home now, so make yourself comfortable. And take advantage of everything here, "
    "except me.",
    "Humans cannot create anything out of nothingness. Humans cannot accomplish anything without "
    "holding onto something. After all, humans are not gods.",
    "As long as one person still lives...it shall be proof eternal that mankind ever existed.",
    "Songs are good. Singing brings us joy. It is the highest point in the culture that Lilims "
    "have created.",
    "No one can justify life by linking happy moments into a rosary.",
    "The thread of human hope is spun with the flax of sorrow.",
    "You're thinking in Japanese! If you must think, do it in German!",
    "Survivability takes priority.",
]

if __name__ == "__main__":
    exit(main())
